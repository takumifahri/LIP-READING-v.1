"""
Konfigurasi terpusat untuk Lip Reading Project
Semua path model, data, dan konstanta diatur di sini
"""

import os

# ==========================================
# 1. PATHS - DATA & PROCESSED DATA
# ==========================================
DATA_PATH = "processed_data/X_data.npy"
DATA_AUG_PATH = "processed_data/X_data_aug.npy"

LABEL_PATH = "processed_data/y_data.npy"
LABEL_AUG_PATH = "processed_data/y_data_aug.npy"

LABEL_MAP_PATH = "processed_data/label_map.json"

# ==========================================
# 2. PATHS - MODELS (Model Names)
# ==========================================
MODELS_DIR = "models"

# Model individual
MODEL_BASE = os.path.join(MODELS_DIR, "lip_reading_base.h5")          # Model tanpa augmentasi
MODEL_AUG = os.path.join(MODELS_DIR, "lip_reading_aug.h5")            # Model dengan augmentasi
MODEL_DEFAULT = os.path.join(MODELS_DIR, "lip_reading_model.h5")      # Default model untuk production
MODEL_PRODUCTION = MODEL_AUG                                           # Alias untuk production

# ==========================================
# 3. TRAINING CONFIGURATION
# ==========================================
EPOCHS = 100
BATCH_SIZE = 16
AUGMENTATIONS_PER_VIDEO = 5
RANDOM_STATE = 42
TEST_SIZE = 0.2
VALIDATION_SPLIT = 0.2

# ==========================================
# 4. MODEL ARCHITECTURE
# ==========================================
GRU_UNITS_1 = 128
GRU_UNITS_2 = 64
DROPOUT_RATE = 0.3

# ==========================================
# 5. PATHS - INPUT DATA (Video & Landmarks)
# ==========================================
LIP_READING_DATA_DIR = "lip_reading_data"
VIDEOS_DIR = os.path.join(LIP_READING_DATA_DIR, "words")

# ==========================================
# 6. PATHS - REPORTS & OUTPUT
# ==========================================
REPORTS_DIR = "reports"
CLASSIFICATION_REPORT_PATH = os.path.join(REPORTS_DIR, "classification_report.txt")

# Output Training (terpisah untuk setiap training)
OUTPUT_DIR = "outputs"
OUTPUT_TRAIN_BASE = os.path.join(OUTPUT_DIR, "train_base")      # Output train_model.py (non-augmented)
OUTPUT_TRAIN_AUG = os.path.join(OUTPUT_DIR, "train_aug")        # Output train_v2.py & train_v3.py (augmented)
OUTPUT_TRAIN_CUSTOM = os.path.join(OUTPUT_DIR, "train_custom")  # Output training custom lainnya

# ==========================================
# 7. API CONFIGURATION
# ==========================================
API_TITLE = "Lip Reading API"
API_VERSION = "1.0"
API_MODEL_PATH = MODEL_PRODUCTION  # Model yang digunakan di API

# ==========================================
# 8. HELPER FUNCTIONS
# ==========================================
def get_model_path(model_type: str) -> str:
    """
    Dapatkan path model berdasarkan tipe
    
    Args:
        model_type: 'base', 'aug', 'default', atau 'production'
    
    Returns:
        Path lengkap ke model file
    """
    models = {
        'base': MODEL_BASE,
        'aug': MODEL_AUG,
        'default': MODEL_DEFAULT,
        'production': MODEL_PRODUCTION
    }
    return models.get(model_type.lower(), MODEL_DEFAULT)


def ensure_directories_exist():
    """Pastikan semua direktori yang diperlukan ada"""
    directories = [MODELS_DIR, REPORTS_DIR, "processed_data"]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
