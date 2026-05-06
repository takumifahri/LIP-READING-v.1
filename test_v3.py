import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import tensorflow as tf
import json
from collections import deque
import os
import time

# ==========================================
# 1. KONFIGURASI & AMBANG BATAS (V3)
# ==========================================
MODEL_FILENAME = "lip_reading_model.h5"
MODEL_PATH = os.path.join("models", MODEL_FILENAME)
LABEL_MAP_PATH = "processed_data/label_map.json"
FACE_LANDMARKER_TASK = "face_landmarker.task"

# --- AMBANG BATAS ANTI-GUESSING ---
# 1. Lip Opening Threshold: Berapa lebar minimal bibir terbuka (0.01 - 0.05)
LIP_OPENING_THRESHOLD = 0.025 

# 2. Activity Threshold (Movement): Standar deviasi gerakan bibir dalam window
# Ini mendeteksi apakah bibir BENAR-BENAR BERGERAK (ngomong) atau cuma diam/jitter
ACTIVITY_THRESHOLD = 0.003 

# 3. Confidence Threshold: Minimal keyakinan model (90%)
CONFIDENCE_THRESHOLD = 0.90 

# 4. Prediction Stability: Harus berapa kali kata yang sama muncul berturut-turut
PREDICTION_STABILITY = 4 

# 5. Window & Sequence
MAX_SEQUENCE_FRAMES = 90
results_buffer = deque(maxlen=15) # Buffer untuk voting kata

# Indeks bibir MediaPipe
LIP_INDICES = [
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95,
    78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308
]

# ==========================================
# 2. INISIALISASI MODEL & DETECTOR
# ==========================================
os.environ["QT_QPA_PLATFORM"] = "xcb"

if not os.path.exists(MODEL_PATH):
    print(f"Error: Model {MODEL_PATH} tidak ditemukan!")
    exit()

print("⌛ Loading Model...")
model = tf.keras.models.load_model(MODEL_PATH)

with open(LABEL_MAP_PATH, 'r') as f:
    label_map = json.load(f)
target_names = [word for word, idx in sorted(label_map.items(), key=lambda item: item[1])]

base_options = python.BaseOptions(model_asset_path=FACE_LANDMARKER_TASK)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=False,
    num_faces=1
)
detector = vision.FaceLandmarker.create_from_options(options)

# ==========================================
# 3. FUNGSI PEMBANTU
# ==========================================
def calculate_lip_distance(landmarks):
    # Landmark 13 (bibir atas dalam), Landmark 14 (bibir bawah dalam)
    p13 = landmarks[13]
    p14 = landmarks[14]
    # Landmark 10 (dahi atas), Landmark 152 (dagu) untuk normalisasi face height
    p10 = landmarks[10]
    p152 = landmarks[152]
    
    face_height = np.sqrt((p10.x - p152.x)**2 + (p10.y - p152.y)**2)
    dist = np.sqrt((p13.x - p14.x)**2 + (p13.y - p14.y)**2)
    
    # Kembalikan jarak yang sudah dinormalisasi terhadap ukuran wajah
    return dist / face_height if face_height > 0 else 0

# ==========================================
# 4. MAIN LOOP
# ==========================================
cap = cv2.VideoCapture(0)
sequence = deque(maxlen=MAX_SEQUENCE_FRAMES)
lip_distance_history = deque(maxlen=20) # Untuk cek aktivitas/pergerakan

last_stable_word = "..."
last_prediction_time = 0
is_currently_speaking = False

print("\n🚀 Lip Reading V3 - Anti-Guessing Engine")
print("Mode: Robust Activity Detection\n")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    detection_result = detector.detect(mp_image)
    
    display_frame = frame.copy()
    current_dist = 0
    
    if detection_result.face_landmarks:
        landmarks = detection_result.face_landmarks[0]
        
        # 1. Hitung Lip Opening (Normalisasi ke wajah)
        current_dist = calculate_lip_distance(landmarks)
        lip_distance_history.append(current_dist)
        
        # 2. Ekstrak Landmark Bibir untuk Model
        lip_points = []
        for idx in LIP_INDICES:
            pt = landmarks[idx]
            lip_points.append([pt.x, pt.y])
            # Draw landmarks
            cv2.circle(display_frame, (int(pt.x * frame.shape[1]), int(pt.y * frame.shape[0])), 2, (0, 255, 0), -1)
        
        # Normalisasi titik bibir (Sesuai cara training)
        lip_points = np.array(lip_points)
        center = np.mean(lip_points, axis=0)
        lip_points = lip_points - center
        max_dist = np.max(np.abs(lip_points))
        if max_dist > 0: lip_points = lip_points / max_dist
        
        sequence.append(lip_points.flatten())

    # --- LOGIKA PREDIKSI & ANTI-GUESSING ---
    if len(sequence) == MAX_SEQUENCE_FRAMES:
        # A. Cek Aktivitas (Apakah bibir bergerak/berbicara dalam window terakhir)
        # Hitung variasi jarak bibir (std dev) dalam 20 frame terakhir
        recent_activity = np.std(list(lip_distance_history)) if len(lip_distance_history) > 0 else 0
        
        # Syarat Berbicara: 
        # 1. Jarak bibir saat ini > Threshold
        # 2. Ada pergerakan (std dev > Activity Threshold)
        is_speaking = current_dist > LIP_OPENING_THRESHOLD or recent_activity > ACTIVITY_THRESHOLD
        
        if is_speaking:
            is_currently_speaking = True
            # Lakukan prediksi hanya jika sedang aktif
            input_data = np.expand_dims(list(sequence), axis=0)
            prediction = model.predict(input_data, verbose=0)
            confidence = np.max(prediction)
            predicted_idx = np.argmax(prediction)
            word = target_names[predicted_idx]
            
            # Hanya simpan jika sangat yakin
            if confidence > CONFIDENCE_THRESHOLD:
                results_buffer.append(word)
            
            # Voting untuk stabilitas
            if len(results_buffer) > 0:
                most_common = max(set(results_buffer), key=list(results_buffer).count)
                if list(results_buffer).count(most_common) >= PREDICTION_STABILITY:
                    last_stable_word = most_common
                    last_prediction_time = time.time()
            
            status_text = f"KATA: {last_stable_word.upper()} ({confidence*100:.1f}%)"
            status_color = (0, 255, 0)
        else:
            # Jika diam, beri jeda 1 detik sebelum tulisan hilang (UX)
            if time.time() - last_prediction_time > 1.5:
                last_stable_word = "..."
                results_buffer.clear()
                is_currently_speaking = False
            
            status_text = "STATUS: DIAM / IDLE"
            status_color = (255, 255, 0)

        # --- UI OVERLAY ---
        # 1. Bar Background
        cv2.rectangle(display_frame, (10, 20), (600, 85), (0,0,0), -1)
        cv2.putText(display_frame, status_text, (25, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, status_color, 2)
        
        # 2. Indikator Debug (Thresholds)
        # Lip Opening Bar
        cv2.putText(display_frame, f"Opening: {current_dist:.4f}", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        open_bar_w = int(min(current_dist * 2000, 200))
        cv2.rectangle(display_frame, (20, 120), (220, 130), (50, 50, 50), -1)
        cv2.rectangle(display_frame, (20, 120), (20 + open_bar_w, 130), status_color, -1)
        cv2.line(display_frame, (20 + int(LIP_OPENING_THRESHOLD*2000), 115), (20 + int(LIP_OPENING_THRESHOLD*2000), 135), (0, 0, 255), 2)
        
        # Activity (Movement) Bar
        cv2.putText(display_frame, f"Activity: {recent_activity:.4f}", (250, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        act_bar_w = int(min(recent_activity * 20000, 200))
        cv2.rectangle(display_frame, (250, 120), (450, 130), (50, 50, 50), -1)
        cv2.rectangle(display_frame, (250, 120), (250 + act_bar_w, 130), (255, 0, 255), -1)
        cv2.line(display_frame, (250 + int(ACTIVITY_THRESHOLD*20000), 115), (250 + int(ACTIVITY_THRESHOLD*20000), 135), (0, 0, 255), 2)

    else:
        # Loading frames...
        p = int(len(sequence)/MAX_SEQUENCE_FRAMES * 100)
        cv2.putText(display_frame, f"MEMUAT BUFFER: {p}%", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow("Anti-Gravity Lip Reading V3", display_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
