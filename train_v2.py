import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import Sequence
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
import json

# Konfigurasi
DATA_PATH = "processed_data/X_data_aug.npy"
LABEL_PATH = "processed_data/y_data_aug.npy"
MAP_PATH = "processed_data/label_map.json"
MODEL_SAVE_PATH = "models/lip_reading_aug.h5"

class LipReadingDataGenerator(Sequence):
    """
    Generator untuk melakukan augmentasi data landmark bibir on-the-fly.
    Membantu model menjadi lebih robust terhadap variasi posisi dan kecepatan.
    """
    def __init__(self, x_set, y_set, batch_size=16, augment=False):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size
        self.augment = augment

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        if self.augment:
            # Terapkan augmentasi pada setiap sampel dalam batch
            batch_x = np.array([self._apply_augmentation(x) for x in batch_x])
        
        return batch_x, batch_y

    def _apply_augmentation(self, data):
        # Data shape: (frames, 64) -> Diambil dari 32 titik * 2 (x,y)
        frames, features = data.shape
        landmarks = data.reshape(frames, 32, 2)
        
        # 1. Random Rotation (-5 sampai 5 derajat)
        if np.random.random() > 0.5:
            theta = np.radians(np.random.uniform(-5, 5))
            c, s = np.cos(theta), np.sin(theta)
            R = np.array(((c, -s), (s, c)))
            landmarks = np.dot(landmarks, R)
            
        # 2. Random Scaling (0.95 sampai 1.05)
        if np.random.random() > 0.5:
            scale = np.random.uniform(0.95, 1.05)
            landmarks *= scale
            
        # 3. Gaussian Noise (Simulasi jitter sensor/kamera)
        if np.random.random() > 0.5:
            noise = np.random.normal(0, 0.0015, landmarks.shape)
            landmarks += noise

        # 4. Temporal Jitter (Simulasi variasi kecepatan bicara)
        # Menghapus 2 frame acak dan menduplikat frame terakhir untuk jaga panjang tetap
        if np.random.random() > 0.7:
            idx_to_keep = np.sort(np.random.choice(frames, frames-2, replace=False))
            landmarks_jittered = landmarks[idx_to_keep]
            padding = np.repeat(landmarks[-1:], 2, axis=0)
            landmarks = np.concatenate([landmarks_jittered, padding], axis=0)

        return landmarks.reshape(frames, features)

def train():
    # 1. Load Data
    if not os.path.exists(DATA_PATH):
        print(f"Error: File {DATA_PATH} tidak ditemukan. Jalankan extract_features.py dulu!")
        return

    X = np.load(DATA_PATH)
    y = np.load(LABEL_PATH)
    
    with open(MAP_PATH, 'r') as f:
        label_map = json.load(f)
    
    num_classes = len(label_map)
    print(f"✓ Data loaded. Shape X: {X.shape}, Total Classes: {num_classes}")

    # 2. Splitting Data (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"✓ Splitting selesai. Train: {len(X_train)}, Test: {len(X_test)}")

    # Inisialisasi Data Generators
    train_gen = LipReadingDataGenerator(X_train, y_train, batch_size=16, augment=True)
    val_gen = LipReadingDataGenerator(X_test, y_test, batch_size=16, augment=False)

    # 3. Build Model (GRU Architecture)
    model = Sequential([
        GRU(128, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
        Dropout(0.3),
        BatchNormalization(),
        
        GRU(64, return_sequences=False),
        Dropout(0.3),
        BatchNormalization(),
        
        Dense(64, activation='relu'),
        Dropout(0.2),
        
        Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    model.summary()

    # 4. Training
    print("\n🚀 Memulai training dengan Data Augmentation...")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=150,
        verbose=1
    )

    # 5. Save Model
    os.makedirs("models", exist_ok=True)
    model.save(MODEL_SAVE_PATH)
    print(f"\n✅ Model berhasil disimpan di: {MODEL_SAVE_PATH}")

    # 6. Visualisasi
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Loss')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    train()
