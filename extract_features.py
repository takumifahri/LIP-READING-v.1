import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm
from config import VIDEOS_DIR, LIP_READING_DATA_DIR

# Konfigurasi
DATASET_DIR = Path(VIDEOS_DIR)
OUTPUT_DIR = Path("processed_data")
OUTPUT_DIR.mkdir(exist_ok=True)
MODEL_PATH = 'face_landmarker.task'

# MediaPipe Tasks Setup
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False,
    num_faces=1
)
detector = vision.FaceLandmarker.create_from_options(options)

# Indeks bibir MediaPipe (Tetap sama)
LIP_INDICES = [
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95,
    78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308
]

def extract_landmarks_from_video(video_path, max_frames=90):
    cap = cv2.VideoCapture(str(video_path))
    sequence = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # Detect landmarks
        detection_result = detector.detect(mp_image)
        
        if detection_result.face_landmarks:
            landmarks = detection_result.face_landmarks[0]
            
            # Ambil koordinat bibir (X, Y)
            lip_points = []
            for idx in LIP_INDICES:
                pt = landmarks[idx]
                lip_points.append([pt.x, pt.y])
            
            lip_points = np.array(lip_points)
            
            # --- NORMALISASI ---
            # 1. Zero-centering
            center = np.mean(lip_points, axis=0)
            lip_points = lip_points - center
            
            # 2. Scaling
            max_dist = np.max(np.abs(lip_points))
            if max_dist > 0:
                lip_points = lip_points / max_dist
            
            sequence.append(lip_points.flatten())
            
        if len(sequence) >= max_frames:
            break
            
    cap.release()
    
    # Padding
    if len(sequence) < max_frames:
        if len(sequence) == 0: return None
        padding = [sequence[-1]] * (max_frames - len(sequence))
        sequence.extend(padding)
        
    return np.array(sequence)

def process_all_data():
    X, y = [], []
    words = sorted([d.name for d in DATASET_DIR.iterdir() if d.is_dir()])
    label_map = {word: i for i, word in enumerate(words)}
    
    print(f"Target: {len(words)} kata. Memulai ekstraksi...")
    
    for word in words:
        video_files = list((DATASET_DIR / word / "videos").glob("*.mp4"))
        if not video_files: continue
        
        print(f"Processing: {word}")
        for v_path in tqdm(video_files):
            features = extract_landmarks_from_video(v_path)
            if features is not None:
                X.append(features)
                y.append(label_map[word])
    
    X, y = np.array(X), np.array(y)
    np.save(OUTPUT_DIR / "X_data.npy", X)
    np.save(OUTPUT_DIR / "y_data.npy", y)
    
    import json
    with open(OUTPUT_DIR / "label_map.json", "w") as f:
        json.dump(label_map, f)
    
    print(f"\nBerhasil! Shape X: {X.shape}")

if __name__ == "__main__":
    process_all_data()
