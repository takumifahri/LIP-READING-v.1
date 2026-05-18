import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout, BatchNormalization
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
import json
from config import (
    DATA_PATH, LABEL_PATH, LABEL_MAP_PATH, MODEL_BASE,
    EPOCHS, BATCH_SIZE, RANDOM_STATE, TEST_SIZE, VALIDATION_SPLIT,
    GRU_UNITS_1, GRU_UNITS_2, DROPOUT_RATE, ensure_directories_exist,
    OUTPUT_TRAIN_BASE
)

# Gunakan konfigurasi dari config.py
MODEL_SAVE_PATH = MODEL_BASE
MODEL_LABEL = "base"
MAP_PATH = LABEL_MAP_PATH

# Folder output training (terpisah)
TRAIN_OUTPUT_DIR = OUTPUT_TRAIN_BASE

def ensure_output_dir():
    """Pastikan folder output ada"""
    os.makedirs(TRAIN_OUTPUT_DIR, exist_ok=True)
    os.makedirs("models", exist_ok=True)

def train():
    ensure_output_dir()
    
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
    ], name=f"lip_reading_{MODEL_LABEL}_gru")

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("\n📋 Model Summary:")
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
    model.save(MODEL_SAVE_PATH)
    print(f"\n✅ Model berhasil disimpan di: {MODEL_SAVE_PATH}")

    # 6. Visualisasi Training History & Simpan
    plt.figure(figsize=(14, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
    plt.plot(history.history['val_accuracy'], label='Val Accuracy', linewidth=2)
    plt.title('Model Accuracy (Base/Non-Augmented)', fontsize=12, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss', linewidth=2)
    plt.plot(history.history['val_loss'], label='Val Loss', linewidth=2)
    plt.title('Model Loss (Base/Non-Augmented)', fontsize=12, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    
    # Simpan plot ke folder output
    history_plot_path = os.path.join(TRAIN_OUTPUT_DIR, "training_history.png")
    plt.savefig(history_plot_path, dpi=150, bbox_inches='tight')
    print(f"📊 Training history disimpan ke: {history_plot_path}")
    
    # Simpan training metrics
    metrics_path = os.path.join(TRAIN_OUTPUT_DIR, "training_metrics.txt")
    with open(metrics_path, 'w') as f:
        f.write("=" * 50 + "\n")
        f.write("TRAINING METRICS - BASE MODEL (NON-AUGMENTED)\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Data: {DATA_PATH}\n")
        f.write(f"Model: {MODEL_SAVE_PATH}\n")
        f.write(f"Train samples: {len(X_train)}\n")
        f.write(f"Test samples: {len(X_test)}\n")
        f.write(f"Number of classes: {num_classes}\n")
        f.write(f"Final Train Accuracy: {history.history['accuracy'][-1]:.4f}\n")
        f.write(f"Final Val Accuracy: {history.history['val_accuracy'][-1]:.4f}\n")
        f.write(f"Final Train Loss: {history.history['loss'][-1]:.4f}\n")
        f.write(f"Final Val Loss: {history.history['val_loss'][-1]:.4f}\n")
    
    print(f"📝 Training metrics disimpan ke: {metrics_path}")
    
    plt.show()

if __name__ == "__main__":
    train()
