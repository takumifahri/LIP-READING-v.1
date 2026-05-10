import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import Sequence
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
import json

# ==========================================
# 1. KONFIGURASI
# ==========================================
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
            batch_x = np.array([self._apply_augmentation(x) for x in batch_x])
        
        return batch_x, batch_y

    def _apply_augmentation(self, data):
        frames, features = data.shape
        landmarks = data.reshape(frames, 32, 2)
        
        # Rotation
        if np.random.random() > 0.5:
            theta = np.radians(np.random.uniform(-5, 5))
            c, s = np.cos(theta), np.sin(theta)
            R = np.array(((c, -s), (s, c)))
            landmarks = np.dot(landmarks, R)
            
        # Scaling
        if np.random.random() > 0.5:
            scale = np.random.uniform(0.95, 1.05)
            landmarks *= scale
            
        # Noise
        if np.random.random() > 0.5:
            noise = np.random.normal(0, 0.001, landmarks.shape)
            landmarks += noise

        return landmarks.reshape(frames, features)

def train():
    # 1. Load Data
    if not os.path.exists(DATA_PATH):
        print(f"Error: File {DATA_PATH} tidak ditemukan!")
        return

    X = np.load(DATA_PATH)
    y = np.load(LABEL_PATH)
    
    with open(MAP_PATH, 'r') as f:
        label_map = json.load(f)
    
    num_classes = len(label_map)
    print(f"✓ Data loaded. X: {X.shape}, Classes: {num_classes}")

    # 2. Splitting (85% Train, 15% Validation)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )

    train_gen = LipReadingDataGenerator(X_train, y_train, batch_size=32, augment=True)
    val_gen = LipReadingDataGenerator(X_val, y_val, batch_size=32, augment=False)

    # 3. Model Architecture
    model = Sequential([
        GRU(128, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
        Dropout(0.4),
        BatchNormalization(),
        
        GRU(64, return_sequences=False),
        Dropout(0.4),
        BatchNormalization(),
        
        Dense(64, activation='relu'),
        Dropout(0.3),
        
        Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # 4. OPTIMASI CALLBACKS (Kunci Perbaikan)
    # Berhenti jika val_loss tidak turun selama 15 epoch
    early_stop = EarlyStopping(
        monitor='val_loss', 
        patience=15, 
        restore_best_weights=True,
        verbose=1
    )

    # Simpan model terbaik secara otomatis
    checkpoint = ModelCheckpoint(
        MODEL_SAVE_PATH,
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )

    # Kurangi Learning Rate jika stagnan
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=0.00001,
        verbose=1
    )

    # 5. Training Execution
    print("\n🚀 Memulai Optimized Training...")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=150,
        callbacks=[early_stop, checkpoint, reduce_lr],
        verbose=1
    )

    print(f"\n✅ Training selesai! Model terbaik disimpan di: {MODEL_SAVE_PATH}")

    # 6. Plotting
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.legend(); plt.title('Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.legend(); plt.title('Loss')
    plt.show()

if __name__ == "__main__":
    train()
