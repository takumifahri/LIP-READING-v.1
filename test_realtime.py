import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import tensorflow as tf
import json
from collections import deque
import os
from config import MODEL_DEFAULT, LABEL_MAP_PATH

# 1. KONFIGURASI
# Gunakan model hasil training terbaru
MODEL_PATH = MODEL_DEFAULT

# Fix untuk error Qt di Linux/Wayland
os.environ["QT_QPA_PLATFORM"] = "xcb"
LABEL_MAP_PATH = LABEL_MAP_PATH
FACE_LANDMARKER_TASK = "face_landmarker.task"
MOVEMENT_THRESHOLD = 0.05  # Sensitivitas gerakan (0.05 - 0.15)
PREDICTION_STABILITY_THRESHOLD = 3  # Berapa kali kata harus muncul berturut-turut agar ditampilkan
CONFIDENCE_THRESHOLD = 0.75  # Tingkat keyakinan minimal model (0.0 - 1.0)

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
sequence = deque(maxlen=90)
results_buffer = deque(maxlen=PREDICTION_STABILITY_THRESHOLD) # Antrian untuk stabilisasi prediksi
last_stable_word = "..." # Menyimpan kata terakhir yang stabil

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
        # --- CEK APAKAH BIBIR BERGERAK ---
        # Hitung standar deviasi gerakan dalam sequence
        movement_amount = np.std(np.array(sequence))
        
        if movement_amount > MOVEMENT_THRESHOLD:
            input_data = np.expand_dims(list(sequence), axis=0)
            prediction = model.predict(input_data, verbose=0)
            predicted_idx = np.argmax(prediction)
            confidence = np.max(prediction)
            word = target_names[predicted_idx]
            
            # Masukkan hasil ke buffer untuk stabilisasi
            if confidence > CONFIDENCE_THRESHOLD:
                results_buffer.append(word)
            
            # Cek apakah buffer sudah konsisten menebak kata yang sama
            if len(results_buffer) >= PREDICTION_STABILITY_THRESHOLD:
                # Ambil kata yang paling sering muncul di buffer
                most_common = max(set(results_buffer), key=list(results_buffer).count)
                if list(results_buffer).count(most_common) >= PREDICTION_STABILITY_THRESHOLD:
                    last_stable_word = most_common
            
            color = (0, 255, 0) if confidence > CONFIDENCE_THRESHOLD else (0, 0, 255)
            text = f"KATA: {last_stable_word.upper()} ({confidence*100:.1f}%)"
        else:
            text = "STATUS: DIAM / IDLE"
            results_buffer.clear() # Reset buffer saat diam
            last_stable_word = "..."
            color = (255, 255, 0)
        
        cv2.rectangle(display_frame, (10, 20), (550, 70), (0,0,0), -1) # Background text
        cv2.putText(display_frame, text, (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        # Tampilkan info movement untuk debugging (bisa dihapus nanti)
        cv2.putText(display_frame, f"Move: {movement_amount:.4f}", (20, 100), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
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
