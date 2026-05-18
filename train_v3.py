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
from config import (
    DATA_AUG_PATH, LABEL_AUG_PATH, LABEL_MAP_PATH, MODEL_AUG,
    EPOCHS, BATCH_SIZE, AUGMENTATIONS_PER_VIDEO, RANDOM_STATE,
    TEST_SIZE, VALIDATION_SPLIT, GRU_UNITS_1, GRU_UNITS_2, DROPOUT_RATE,
    ensure_directories_exist, OUTPUT_TRAIN_AUG
)

# ==========================================
# 1. KONFIGURASI
# ==========================================
DATA_PATH = DATA_AUG_PATH
LABEL_PATH = LABEL_AUG_PATH
MAP_PATH = LABEL_MAP_PATH
MODEL_SAVE_PATH = MODEL_AUG
MODEL_LABEL = "aug"

# Folder output training (terpisah untuk augmented)
TRAIN_OUTPUT_DIR = OUTPUT_TRAIN_AUG

def ensure_output_dir():
    """Pastikan folder output ada"""
    os.makedirs(TRAIN_OUTPUT_DIR, exist_ok=True)
    os.makedirs("models", exist_ok=True)

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
    ensure_output_dir()
    
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

    # 2. Split per video asli, bukan per sampel augmentasi.
    # Urutan X_data_aug.npy dari extract_features_v2.py adalah:
    # original, slow_mo, rotate -10, rotate +10, zoom untuk tiap video.
    # Kalau split dilakukan langsung pada 1750 sampel, varian dari video yang
    # sama bisa masuk train dan validation sekaligus. Itu data leakage.
    if len(X) % AUGMENTATIONS_PER_VIDEO != 0:
        raise ValueError(
            "Jumlah sampel augmented tidak habis dibagi "
            f"{AUGMENTATIONS_PER_VIDEO}. Jalankan ulang extract_features_v2.py "
            "atau gunakan metadata grup sebelum training."
        )

    original_indices = np.arange(len(X) // AUGMENTATIONS_PER_VIDEO)
    original_labels = y[::AUGMENTATIONS_PER_VIDEO]
    train_original, val_original = train_test_split(
        original_indices,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=original_labels,
    )

    train_idx = np.concatenate([
        np.arange(i * AUGMENTATIONS_PER_VIDEO, (i + 1) * AUGMENTATIONS_PER_VIDEO)
        for i in train_original
    ])
    # Validasi paling jujur memakai rekaman original yang belum pernah dilihat,
    # bukan varian augmentasi dari rekaman validation.
    val_idx = val_original * AUGMENTATIONS_PER_VIDEO

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    print(
        "✓ Group split selesai. "
        f"Train original videos: {len(train_original)}, Val original videos: {len(val_original)}"
    )
    print(f"✓ Train samples: {len(X_train)}, Validation original samples: {len(X_val)}")

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
    ], name=f"lip_reading_{MODEL_LABEL}_gru")

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

    # 6. Plotting & Save Output
    plt.figure(figsize=(14, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
    plt.plot(history.history['val_accuracy'], label='Val Accuracy', linewidth=2)
    plt.title('Model Accuracy (Augmented with Early Stopping)', fontsize=12, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss', linewidth=2)
    plt.plot(history.history['val_loss'], label='Val Loss', linewidth=2)
    plt.title('Model Loss (Augmented with Early Stopping)', fontsize=12, fontweight='bold')
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
        f.write("TRAINING METRICS - AUGMENTED MODEL (V3 with Callbacks)\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Data: {DATA_PATH}\n")
        f.write(f"Model: {MODEL_SAVE_PATH}\n")
        f.write(f"Train videos (original): {len(train_original)}\n")
        f.write(f"Train samples (augmented): {len(X_train)}\n")
        f.write(f"Val samples: {len(X_val)}\n")
        f.write(f"Number of classes: {num_classes}\n")
        f.write(f"Augmentations per video: {AUGMENTATIONS_PER_VIDEO}\n")
        f.write(f"Final Train Accuracy: {history.history['accuracy'][-1]:.4f}\n")
        f.write(f"Final Val Accuracy: {history.history['val_accuracy'][-1]:.4f}\n")
        f.write(f"Final Train Loss: {history.history['loss'][-1]:.4f}\n")
        f.write(f"Final Val Loss: {history.history['val_loss'][-1]:.4f}\n")
        f.write(f"Total Epochs Trained: {len(history.history['loss'])}\n")
    
    print(f"📝 Training metrics disimpan ke: {metrics_path}")
    
    plt.show()

if __name__ == "__main__":
    train()
