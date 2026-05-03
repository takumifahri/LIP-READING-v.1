import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import tensorflow as tf
import json
from collections import deque
import os

# 1. KONFIGURASI
# Menggunakan nama model terbaru yang ada di folder models
MODEL_FILENAME = "model_20260503_091431.h5"
MODEL_PATH = os.path.join("models", MODEL_FILENAME)
LABEL_MAP_PATH = "processed_data/label_map.json"
FACE_LANDMARKER_TASK = "face_landmarker.task"

# Indeks bibir MediaPipe
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
    output_facial_transformation_matrixes=False,
    num_faces=1
)
detector = vision.FaceLandmarker.create_from_options(options)

# 4. REAL-TIME PREDICTION
cap = cv2.VideoCapture(0)
sequence = deque(maxlen=90) # Kita pakai 90 frame (3 detik) sesuai data training

print("\n🚀 Real-time Testing Dimulai!")
print("Instruksi: Gerakkan bibirmu di depan kamera. Prediksi akan muncul setelah 3 detik awal.")
print("Tekan 'q' untuk keluar.\n")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    
    # Flip frame biar kayak cermin
    frame = cv2.flip(frame, 1)
    
    # Preprocessing frame untuk MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    detection_result = detector.detect(mp_image)
    
    display_frame = frame.copy()
    
    if detection_result.face_landmarks:
        landmarks = detection_result.face_landmarks[0]
        lip_points = []
        
        # Ekstrak koordinat bibir
        for idx in LIP_INDICES:
            pt = landmarks[idx]
            lip_points.append([pt.x, pt.y])
            # Visualisasi titik bibir di layar
            cv2.circle(display_frame, (int(pt.x * frame.shape[1]), int(pt.y * frame.shape[0])), 2, (0, 255, 0), -1)
        
        # Normalisasi
        lip_points = np.array(lip_points)
        center = np.mean(lip_points, axis=0)
        lip_points = lip_points - center
        max_dist = np.max(np.abs(lip_points))
        if max_dist > 0:
            lip_points = lip_points / max_dist
        
        # Masukkan ke dalam antrian sequence
        sequence.append(lip_points.flatten())

    # Jika antrian sudah penuh, mulai prediksi
    if len(sequence) == 90:
        input_data = np.expand_dims(list(sequence), axis=0)
        prediction = model.predict(input_data, verbose=0)
        predicted_idx = np.argmax(prediction)
        confidence = np.max(prediction)
        
        word = target_names[predicted_idx]
        
        # UI Feedback
        color = (0, 255, 0) if confidence > 0.6 else (0, 0, 255)
        text = f"KATA: {word.upper()} ({confidence*100:.1f}%)"
        cv2.rectangle(display_frame, (10, 20), (500, 70), (0,0,0), -1) # Background text
        cv2.putText(display_frame, text, (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    else:
        # Progress bar untuk mengumpulkan frame awal
        progress = int((len(sequence) / 90) * 100)
        cv2.putText(display_frame, f"Mengumpulkan frame: {progress}%", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Lip Reading Real-Time Test", display_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("\nSelesai.")
