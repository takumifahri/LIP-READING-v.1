import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm
import json

# ==========================================
# KONFIGURASI
# ==========================================
DATASET_DIR = Path("lip_reading_data/words")
OUTPUT_DIR = Path("processed_data")
OUTPUT_DIR.mkdir(exist_ok=True)
MODEL_PATH = 'face_landmarker.task'
MAX_FRAMES = 90

# MediaPipe Setup
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=False,
    num_faces=1
)
detector = vision.FaceLandmarker.create_from_options(options)

LIP_INDICES = [
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95,
    78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308
]

def get_landmarks(frame):
    """Mendapatkan landmark bibir yang ternormalisasi dari satu frame."""
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    detection_result = detector.detect(mp_image)
    
    if detection_result.face_landmarks:
        landmarks = detection_result.face_landmarks[0]
        lip_points = np.array([[landmarks[idx].x, landmarks[idx].y] for idx in LIP_INDICES])
        
        # Normalisasi
        center = np.mean(lip_points, axis=0)
        lip_points -= center
        max_dist = np.max(np.abs(lip_points))
        if max_dist > 0: lip_points /= max_dist
        
        return lip_points.flatten()
    return None

def process_video_augmented(video_path, slow_mo=False, rotate=0, zoom=1.0):
    """Memproses video dengan berbagai augmentasi visual dan temporal."""
    cap = cv2.VideoCapture(str(video_path))
    raw_frames = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        h, w = frame.shape[:2]
        
        # 1. Zoom Augmentation
        if zoom != 1.0:
            new_w, new_h = int(w * zoom), int(h * zoom)
            frame = cv2.resize(frame, (new_w, new_h))
            if zoom > 1.0:
                # Crop tengah
                start_x, start_y = (new_w - w) // 2, (new_h - h) // 2
                frame = frame[start_y:start_y+h, start_x:start_x+w]
            else:
                # Padding hitam
                pad_x, pad_y = (w - new_w) // 2, (h - new_h) // 2
                frame = cv2.copyMakeBorder(frame, pad_y, h - new_h - pad_y, pad_x, w - new_w - pad_x, cv2.BORDER_CONSTANT)

        # 2. Rotation Augmentation
        if rotate != 0:
            M = cv2.getRotationMatrix2D((w // 2, h // 2), rotate, 1.0)
            frame = cv2.warpAffine(frame, M, (w, h))

        raw_frames.append(frame)
    cap.release()

    if not raw_frames: return None

    # 3. Slow Motion (Temporal Interpolation)
    # Menyisipkan frame 'rata-rata' di antara frame asli
    if slow_mo:
        augmented_frames = []
        for i in range(len(raw_frames) - 1):
            augmented_frames.append(raw_frames[i])
            # Interpolasi linear antara dua frame
            mid_frame = cv2.addWeighted(raw_frames[i], 0.5, raw_frames[i+1], 0.5, 0)
            augmented_frames.append(mid_frame)
        augmented_frames.append(raw_frames[-1])
        raw_frames = augmented_frames

    # Ekstraksi fitur dari frames yang sudah di-augmentasi
    sequence = []
    for f in raw_frames:
        feat = get_landmarks(f)
        if feat is not None:
            sequence.append(feat)
        if len(sequence) >= MAX_FRAMES:
            break
            
    if len(sequence) == 0: return None
    
    # Padding jika frame kurang dari 90
    if len(sequence) < MAX_FRAMES:
        last_val = sequence[-1]
        sequence.extend([last_val] * (MAX_FRAMES - len(sequence)))
        
    return np.array(sequence)

def run_augmented_extraction():
    X, y = [], []
    words = sorted([d.name for d in DATASET_DIR.iterdir() if d.is_dir()])
    label_map = {word: i for i, word in enumerate(words)}
    
    print(f"🚀 Memulai Ekstraksi Augmentasi untuk {len(words)} kata...")
    
    for word in words:
        video_files = list((DATASET_DIR / word / "videos").glob("*.mp4"))
        if not video_files: continue
        
        print(f"\nProcessing: {word}")
        for v_path in tqdm(video_files):
            # --- 1. Original ---
            feat = process_video_augmented(v_path)
            if feat is not None: X.append(feat); y.append(label_map[word])
            
            # --- 2. Slow Motion ---
            feat = process_video_augmented(v_path, slow_mo=True)
            if feat is not None: X.append(feat); y.append(label_map[word])
            
            # --- 3. Rotation (Miring) ---
            for angle in [-10, 10]:
                feat = process_video_augmented(v_path, rotate=angle)
                if feat is not None: X.append(feat); y.append(label_map[word])
                
            # --- 4. Zoom In ---
            feat = process_video_augmented(v_path, zoom=1.2)
            if feat is not None: X.append(feat); y.append(label_map[word])

    X, y = np.array(X), np.array(y)
    
    # Simpan dengan nama berbeda agar tidak menimpa data asli
    np.save(OUTPUT_DIR / "X_data_aug.npy", X)
    np.save(OUTPUT_DIR / "y_data_aug.npy", y)
    
    with open(OUTPUT_DIR / "label_map.json", "w") as f:
        json.dump(label_map, f)
        
    print(f"\n✅ SELESAI!")
    print(f"Total Sampel: {len(X)}")
    print(f"Shape X: {X.shape}")
    print(f"File disimpan di: processed_data/X_data_aug.npy")

if __name__ == "__main__":
    run_augmented_extraction()
