import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import tensorflow as tf
import json
import os
import time
from config import MODEL_DEFAULT, LABEL_MAP_PATH

# ==========================================
# 1. KONFIGURASI
# ==========================================
MODEL_PATH = MODEL_DEFAULT
LABEL_MAP_PATH = LABEL_MAP_PATH
FACE_LANDMARKER_TASK = "face_landmarker.task"

# --- PARAMETER TRIGGER ---
LIP_OPENING_THRESHOLD = 0.025 # Ambang batas mulai bicara
SILENCE_TIMEOUT_FRAMES = 20   # Berapa frame diam sebelum dianggap "selesai ngomong"
MIN_RECORDING_FRAMES = 15     # Minimal frame untuk dianggap satu kata valid
MAX_RECORDING_FRAMES = 90     # Maksimal frame yang dikirim ke model

# Indeks bibir MediaPipe
LIP_INDICES = [
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95,
    78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308
]

# ==========================================
# 2. INISIALISASI
# ==========================================
os.environ["QT_QPA_PLATFORM"] = "xcb"

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
    p13 = landmarks[13]
    p14 = landmarks[14]
    p10 = landmarks[10]
    p152 = landmarks[152]
    face_height = np.sqrt((p10.x - p152.x)**2 + (p10.y - p152.y)**2)
    dist = np.sqrt((p13.x - p14.x)**2 + (p13.y - p14.y)**2)
    return dist / face_height if face_height > 0 else 0

def preprocess_sequence(raw_sequence):
    # Padding atau Slicing ke MAX_RECORDING_FRAMES (90)
    seq = list(raw_sequence)
    if len(seq) < MAX_RECORDING_FRAMES:
        # Padding dengan frame terakhir (sama seperti saat training)
        padding = [seq[-1]] * (MAX_RECORDING_FRAMES - len(seq))
        seq.extend(padding)
    elif len(seq) > MAX_RECORDING_FRAMES:
        # Ambil 90 frame terakhir
        seq = seq[-MAX_RECORDING_FRAMES:]
    return np.array(seq)

# ==========================================
# 4. MAIN LOOP (TRIGGER MODE)
# ==========================================
cap = cv2.VideoCapture(0)
recording_buffer = []
silence_counter = 0
is_recording = False
last_result = "Tunggu Bicara..."
last_confidence = 0

print("\n🎤 Lip Reading - Trigger Mode")
print("Model hanya akan menebak SETELAH kamu selesai bicara.")
print("Tekan 'q' untuk keluar.\n")

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
        current_dist = calculate_lip_distance(landmarks)
        
        # Ekstrak fitur bibir
        lip_points = []
        for idx in LIP_INDICES:
            pt = landmarks[idx]
            lip_points.append([pt.x, pt.y])
            cv2.circle(display_frame, (int(pt.x * frame.shape[1]), int(pt.y * frame.shape[0])), 2, (0, 255, 0), -1)
        
        # Normalisasi (Zero-center & Scaling)
        lip_points = np.array(lip_points)
        center = np.mean(lip_points, axis=0)
        lip_points = lip_points - center
        max_dist = np.max(np.abs(lip_points))
        if max_dist > 0: lip_points = lip_points / max_dist
        
        current_features = lip_points.flatten()

        # --- LOGIKA TRIGGER ---
        if current_dist > LIP_OPENING_THRESHOLD:
            # Mulai atau Lanjut Rekaman
            if not is_recording:
                print("⏺️ Deteksi Bicara: Mulai Merekam...")
                is_recording = True
                recording_buffer = []
            
            is_recording = True
            silence_counter = 0
            recording_buffer.append(current_features)
        else:
            # Bibir tertutup, cek apakah sudah selesai bicara
            if is_recording:
                silence_counter += 1
                recording_buffer.append(current_features) # Tetap rekam sedikit diam di akhir
                
                if silence_counter > SILENCE_TIMEOUT_FRAMES:
                    # SELESAI BICARA -> PROSES PREDIKSI
                    is_recording = False
                    if len(recording_buffer) > MIN_RECORDING_FRAMES:
                        print(f"✅ Selesai: Memproses {len(recording_buffer)} frame...")
                        input_data = preprocess_sequence(recording_buffer)
                        input_data = np.expand_dims(input_data, axis=0)
                        
                        prediction = model.predict(input_data, verbose=0)
                        last_confidence = np.max(prediction)
                        last_result = target_names[np.argmax(prediction)]
                        print(f"➡️ Hasil: {last_result.upper()} ({last_confidence*100:.1f}%)")
                    else:
                        print("⚠️ Terlalu pendek, diabaikan.")
                    
                    recording_buffer = []

    # --- UI OVERLAY ---
    # Status Bar
    status_color = (0, 0, 255) if is_recording else (0, 255, 0)
    status_text = "MEREKAM..." if is_recording else "SIAP (SILAHKAN BICARA)"
    
    cv2.rectangle(display_frame, (10, 20), (620, 130), (0,0,0), -1)
    cv2.putText(display_frame, status_text, (25, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
    
    # Hasil Terakhir
    res_text = f"PREDIKSI: {last_result.upper()} ({last_confidence*100:.1f}%)"
    cv2.putText(display_frame, res_text, (25, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Indikator Lip Opening
    bar_w = int(min(current_dist * 2000, 200))
    cv2.rectangle(display_frame, (400, 45), (600, 60), (50, 50, 50), -1)
    cv2.rectangle(display_frame, (400, 45), (400 + bar_w, 60), status_color, -1)
    cv2.putText(display_frame, f"Open: {current_dist:.3f}", (400, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

    cv2.imshow("Lip Reading Trigger Mode", display_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
