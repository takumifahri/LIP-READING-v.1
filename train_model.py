import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout, BatchNormalization
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
import json

# Konfigurasi
DATA_PATH = "processed_data/X_data.npy"
LABEL_PATH = "processed_data/y_data.npy"
MAP_PATH = "processed_data/label_map.json"
MODEL_SAVE_PATH = "models/lip_reading_base.h5"

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
    # Stratify memastikan setiap kata terwakili di data test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"✓ Splitting selesai. Train: {len(X_train)}, Test: {len(X_test)}")

    # 3. Build Model (GRU Architecture)
    model = Sequential([
        # Layer 1: GRU untuk menangkap urutan temporal
        GRU(128, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
        Dropout(0.3),
        BatchNormalization(),
        
        # Layer 2: GRU kedua untuk ekstraksi fitur lebih dalam
        GRU(64, return_sequences=False),
        Dropout(0.3),
        BatchNormalization(),
        
        # Hidden Layer
        Dense(64, activation='relu'),
        Dropout(0.2),
        
        # Output Layer (Softmax untuk klasifikasi multi-kata)
        Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    model.summary()

    # 4. Training
    print("\n🚀 Memulai training...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=150, # Bisa disesuaikan
        batch_size=16,
        verbose=1
    )

    # 5. Save Model
    os.makedirs("models", exist_ok=True)
    model.save(MODEL_SAVE_PATH)
    print(f"\n✅ Model berhasil disimpan di: {MODEL_SAVE_PATH}")

    # 6. Visualisasi (Opsional)
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
