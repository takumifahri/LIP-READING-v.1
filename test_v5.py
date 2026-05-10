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
# 1. KONFIGURASI & AMBANG BATAS (V5 - Anti Jitter)
# ==========================================
MODEL_FILENAME = "lip_reading_aug.h5"
MODEL_PATH = os.path.join("models", MODEL_FILENAME)
LABEL_MAP_PATH = "processed_data/label_map.json"
FACE_LANDMARKER_TASK = "face_landmarker.task"

# --- AMBANG BATAS (DITUNING UNTUK V5) ---
LIP_START_THRESHOLD = 0.038    # Lebih tinggi untuk mulai (Hysteresis)
LIP_KEEP_THRESHOLD = 0.028     # Lebih rendah untuk lanjut merekam (Biaya tahan)
ACTIVITY_THRESHOLD = 0.0035    # Variasi gerakan minimal (ditingkatkan agar tidak sensitif noise)

EMA_ALPHA = 0.3                # Faktor penghalusan (0.1 - 0.5), semakin kecil semakin halus/lambat
START_DELAY_FRAMES = 3         # Harus aktif selama 3 frame baru mulai rekam (Debounce)
SILENCE_TIMEOUT = 20           # Frame diam sebelum cut (ditingkatkan agar lebih sabar)

MIN_FRAMES = 12                # Minimal frame yang direkam agar tidak noise
MAX_FRAMES = 90                # Maksimal frame (kapasitas model)
COOLDOWN_TIME = 1.2            # Jeda setelah muncul hasil (detik)

# Indeks bibir MediaPipe
LIP_INDICES = [
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95,
    78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308
]

# ==========================================
# 2. INISIALISASI
# ==========================================
os.environ["QT_QPA_PLATFORM"] = "xcb"

if not os.path.exists(MODEL_PATH):
    print(f"Error: Model {MODEL_PATH} tidak ditemukan!")
    exit()

print("⌛ Loading Model & Detector...")
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
    p13, p14 = landmarks[13], landmarks[14]
    p10, p152 = landmarks[10], landmarks[152]
    face_height = np.sqrt((p10.x - p152.x)**2 + (p10.y - p152.y)**2)
    dist = np.sqrt((p13.x - p14.x)**2 + (p13.y - p14.y)**2)
    return dist / face_height if face_height > 0 else 0

def preprocess_sequence(raw_sequence):
    seq = list(raw_sequence)
    # Padding ke 90 frame jika kurang (dengan frame terakhir)
    if len(seq) < MAX_FRAMES:
        last_frame = seq[-1]
        seq.extend([last_frame] * (MAX_FRAMES - len(seq)))
    # Potong jika lebih
    return np.array(seq[:MAX_FRAMES])

# ==========================================
# 4. MAIN LOOP
# ==========================================
cap = cv2.VideoCapture(0)
recording_buffer = []
lip_dist_history = deque(maxlen=20)
smoothed_dist = 0               # Untuk EMA
active_frames_count = 0         # Untuk Debounce
silence_counter = 0
is_recording = False
last_prediction = "..."
last_conf = 0.0
cooldown_until = 0

print("\n🚀 Lip Reading V5 - Anti Jitter Edition")
print("Mode: Intelligent Triggering\n")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    detection_result = detector.detect(mp_image)
    
    display_frame = frame.copy()
    current_raw_dist = 0
    recent_activity = 0
    status_text = "MENCARI WAJAH..."
    status_color = (0, 165, 255) # Orange
    
    if detection_result.face_landmarks:
        landmarks = detection_result.face_landmarks[0]
        current_raw_dist = calculate_lip_distance(landmarks)
        
        # --- EMA Smoothing ---
        smoothed_dist = (EMA_ALPHA * current_raw_dist) + ((1 - EMA_ALPHA) * smoothed_dist)
        lip_dist_history.append(smoothed_dist)
        recent_activity = np.std(list(lip_dist_history)) if len(lip_dist_history) > 10 else 0
        
        # Ekstrak fitur bibir
        lip_points = []
        for idx in LIP_INDICES:
            pt = landmarks[idx]
            lip_points.append([pt.x, pt.y])
            cv2.circle(display_frame, (int(pt.x * frame.shape[1]), int(pt.y * frame.shape[0])), 1, (0, 255, 0), -1)
        
        # Normalisasi
        lip_points = np.array(lip_points)
        center = np.mean(lip_points, axis=0)
        lip_points = (lip_points - center)
        max_dist = np.max(np.abs(lip_points))
        if max_dist > 0: lip_points /= max_dist
        current_features = lip_points.flatten()

        # --- LOGIKA TRIGGER V5 ---
        now = time.time()
        
        # Ambang batas berbeda untuk Start vs Keep (Hysteresis)
        threshold = LIP_KEEP_THRESHOLD if is_recording else LIP_START_THRESHOLD
        is_moving = smoothed_dist > threshold or recent_activity > ACTIVITY_THRESHOLD
        
        if now < cooldown_until:
            status_text = "COOLDOWN..."
            status_color = (150, 150, 150)
            active_frames_count = 0
        elif is_moving:
            active_frames_count += 1
            # Debouncing: Harus aktif minimal START_DELAY_FRAMES secara beruntun
            if active_frames_count >= START_DELAY_FRAMES:
                if not is_recording:
                    is_recording = True
                    recording_buffer = []
                    print("⏺️ Mulai Mendengar...")
                
                silence_counter = 0
                if len(recording_buffer) < MAX_FRAMES:
                    recording_buffer.append(current_features)
                
                status_text = f"MENDENGARKAN... ({len(recording_buffer)} frames)"
                status_color = (0, 0, 255)
            else:
                status_text = "VALIDASI GERAKAN..."
                status_color = (0, 255, 255)
        else:
            active_frames_count = 0 # Reset debounce jika diam
            if is_recording:
                silence_counter += 1
                # Masukkan sedikit frame diam biar natural
                if len(recording_buffer) < MAX_FRAMES:
                    recording_buffer.append(current_features)
                
                # Cek jika sudah cukup lama diam (JEDA)
                if silence_counter > SILENCE_TIMEOUT:
                    is_recording = False
                    if len(recording_buffer) > MIN_FRAMES:
                        print(f"✅ Selesai ({len(recording_buffer)} frames). Memproses...")
                        input_data = preprocess_sequence(recording_buffer)
                        input_data = np.expand_dims(input_data, axis=0)
                        
                        prediction = model.predict(input_data, verbose=0)
                        last_conf = np.max(prediction)
                        last_prediction = target_names[np.argmax(prediction)]
                        print(f"➡️ Hasil: {last_prediction.upper()} ({last_conf*100:.1f}%)")
                        
                        cooldown_until = time.time() + COOLDOWN_TIME
                    else:
                        print("⚠️ Terlalu singkat, diabaikan.")
                        last_prediction = "..."
                        last_conf = 0
                
                status_text = "MENUNGGU JEDA..."
                status_color = (255, 255, 0)
            else:
                status_text = "SIAP (SILAHKAN BICARA)"
                status_color = (0, 255, 0)

    # --- UI OVERLAY ---
    # Background Bar
    cv2.rectangle(display_frame, (10, 20), (620, 140), (0,0,0), -1)
    
    # Status
    cv2.putText(display_frame, status_text, (25, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
    
    # Hasil Terakhir
    res_text = f"HASIL: {last_prediction.upper()} ({last_conf*100:.1f}%)"
    cv2.putText(display_frame, res_text, (25, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Indikator Progress Recording
    if is_recording:
        prog_w = int((len(recording_buffer) / MAX_FRAMES) * 580)
        cv2.rectangle(display_frame, (25, 125), (605, 130), (50, 50, 50), -1)
        cv2.rectangle(display_frame, (25, 125), (25 + prog_w, 130), (0, 0, 255), -1)

    # Debug Bars
    cv2.putText(display_frame, f"Smoothed: {smoothed_dist:.3f}", (450, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    cv2.putText(display_frame, f"Act: {recent_activity:.4f}", (450, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

    cv2.imshow("Lip Reading V5 - Anti Jitter", display_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
