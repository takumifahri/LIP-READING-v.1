import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import tensorflow as tf
import json
from collections import deque
import os
from config import MODEL_DEFAULT, LABEL_MAP_PATH, REPORTS_DIR

os.environ["QT_QPA_PLATFORM"] = "xcb"

# Gunakan konfigurasi dari config.py
MODEL_PATH = MODEL_DEFAULT
LABEL_MAP_PATH = LABEL_MAP_PATH
FACE_LANDMARKER_TASK = "face_landmarker.task"

# --- AMBANG BATAS BARU ---
LIP_OPENING_THRESHOLD = 0.015 # Seberapa lebar bibir terbuka (normalisasi 0-1)
CONFIDENCE_THRESHOLD = 0.90   # Naikkan ke 90% agar tidak gampang nebak
PREDICTION_STABILITY = 5      # Harus 5 kali muncul kata yang sama baru tampil

LIP_INDICES = [
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95,
    78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308
]

# 2. LOAD MODEL & LABEL
if not os.path.exists(MODEL_PATH):
    print(f"Error: Model {MODEL_PATH} tidak ditemukan!")
    exit()

print("⌛ Loading Model...")
model = tf.keras.models.load_model(MODEL_PATH)

with open(LABEL_MAP_PATH, 'r') as f:
    label_map = json.load(f)
target_names = [word for word, idx in sorted(label_map.items(), key=lambda item: item[1])]

# 3. SETUP MEDIAPIPE
base_options = python.BaseOptions(model_asset_path=FACE_LANDMARKER_TASK)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=False,
    num_faces=1
)
detector = vision.FaceLandmarker.create_from_options(options)

# 4. REAL-TIME PREDICTION
cap = cv2.VideoCapture(0)
sequence = deque(maxlen=90)
results_buffer = deque(maxlen=20) # Buffer lebih panjang untuk stabilitas
last_stable_word = "..."

print("\n🚀 Real-time Testing Dimulai (Mode Anti-Guessing Aktif)!")
print("Tekan 'q' untuk keluar.\n")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    detection_result = detector.detect(mp_image)
    
    display_frame = frame.copy()
    current_lip_distance = 0
    
    if detection_result.face_landmarks:
        landmarks = detection_result.face_landmarks[0]
        lip_points = []
        
        # --- HITUNG JARAK BIBIR (Lip Opening) ---
        # Landmark 13 (bibir atas dalam), Landmark 14 (bibir bawah dalam)
        p13 = landmarks[13]
        p14 = landmarks[14]
        current_lip_distance = np.sqrt((p13.x - p14.x)**2 + (p13.y - p14.y)**2)

        for idx in LIP_INDICES:
            pt = landmarks[idx]
            lip_points.append([pt.x, pt.y])
            cv2.circle(display_frame, (int(pt.x * frame.shape[1]), int(pt.y * frame.shape[0])), 2, (0, 255, 0), -1)
        
        # Normalisasi
        lip_points = np.array(lip_points)
        center = np.mean(lip_points, axis=0)
        lip_points = lip_points - center
        max_dist = np.max(np.abs(lip_points))
        if max_dist > 0: lip_points = lip_points / max_dist
        
        sequence.append(lip_points.flatten())

    if len(sequence) == 90:
        # --- LOGIKA ANTI-GUESSING ---
        # 1. Cek apakah bibir terbuka cukup lebar (indikasi bicara)
        is_speaking = current_lip_distance > LIP_OPENING_THRESHOLD
        
        if is_speaking:
            input_data = np.expand_dims(list(sequence), axis=0)
            prediction = model.predict(input_data, verbose=0)
            confidence = np.max(prediction)
            predicted_idx = np.argmax(prediction)
            word = target_names[predicted_idx]
            
            # 2. Hanya masukkan ke buffer jika sangat yakin
            if confidence > CONFIDENCE_THRESHOLD:
                results_buffer.append(word)
            
            # 3. Ambil kata yang paling sering muncul di buffer (Voting)
            if len(results_buffer) > 0:
                most_common = max(set(results_buffer), key=list(results_buffer).count)
                if list(results_buffer).count(most_common) >= PREDICTION_STABILITY:
                    last_stable_word = most_common
            
            text = f"KATA: {last_stable_word.upper()} ({confidence*100:.1f}%)"
            color = (0, 255, 0)
        else:
            # Jika diam, reset status secara perlahan
            text = "STATUS: DIAM / IDLE"
            color = (255, 255, 0)
            results_buffer.clear()
            last_stable_word = "..."
        
        # UI Feedback
        cv2.rectangle(display_frame, (10, 20), (580, 75), (0,0,0), -1)
        cv2.putText(display_frame, text, (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        # Tampilkan Indikator Lip Opening untuk Kalibrasi
        cv2.putText(display_frame, f"Lip Opening: {current_lip_distance:.4f}", (20, 100), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.line(display_frame, (20, 110), (20 + int(current_lip_distance*2000), 110), color, 5)

    cv2.imshow("Anti-Guessing Lip Reading", display_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()