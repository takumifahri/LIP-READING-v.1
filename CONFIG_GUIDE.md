# 📋 Konfigurasi Model - Panduan Penggunaan

Semua nama model, path, dan konstanta konfigurasi sekarang **tersentralisasi** di file `config.py`. Ini membuat project lebih mudah dipelihara dan menghindari duplikasi hardcoded values.

---

## 🎯 Daftar Model

Model-model yang tersedia di project:

| Model | Path | Deskripsi |
|-------|------|-----------|
| **Base** | `models/lip_reading_base.h5` | Model tanpa augmentasi data |
| **Aug** | `models/lip_reading_aug.h5` | Model dengan data augmentation |
| **Default** | `models/lip_reading_model.h5` | Model untuk development/testing |
| **Production** | `models/lip_reading_aug.h5` | Alias untuk production (sama dengan Aug) |

---

## 📁 Struktur Config

File `config.py` dibagi menjadi beberapa section:

### 1. **Paths - Data & Processed Data**
```python
DATA_PATH = "processed_data/X_data.npy"
DATA_AUG_PATH = "processed_data/X_data_aug.npy"
LABEL_PATH = "processed_data/y_data.npy"
LABEL_AUG_PATH = "processed_data/y_data_aug.npy"
LABEL_MAP_PATH = "processed_data/label_map.json"
```

### 2. **Paths - Models**
```python
MODEL_BASE = "models/lip_reading_base.h5"
MODEL_AUG = "models/lip_reading_aug.h5"
MODEL_DEFAULT = "models/lip_reading_model.h5"
MODEL_PRODUCTION = MODEL_AUG  # Production menggunakan aug model
```

### 3. **Training Configuration**
```python
EPOCHS = 100
BATCH_SIZE = 16
AUGMENTATIONS_PER_VIDEO = 5
RANDOM_STATE = 42
```

### 4. **API Configuration**
```python
API_TITLE = "Lip Reading API"
API_VERSION = "1.0"
API_MODEL_PATH = MODEL_PRODUCTION  # Model yang digunakan API
```

---

## 🚀 Cara Menggunakan Config

### ✅ Import dari config.py

```python
# Import model path yang dibutuhkan
from config import MODEL_BASE, MODEL_AUG, MODEL_DEFAULT, LABEL_MAP_PATH

# Gunakan di code Anda
model_path = MODEL_AUG
with open(LABEL_MAP_PATH) as f:
    label_map = json.load(f)
```

### ✅ Menggunakan Helper Function

```python
from config import get_model_path

# Dapatkan path model berdasarkan tipe
base_model_path = get_model_path('base')
aug_model_path = get_model_path('aug')
prod_model_path = get_model_path('production')
```

### ✅ Memastikan Direktori Ada

```python
from config import ensure_directories_exist

# Pastikan semua direktori yang diperlukan sudah ada
ensure_directories_exist()
```

---

## 📝 File-File yang Sudah Diupdate

Berikut file-file yang sekarang menggunakan `config.py`:

### Training Files
- ✅ `train_model.py`
- ✅ `train_v2.py`
- ✅ `train_v3.py`

### Testing Files
- ✅ `test2.py`
- ✅ `test_v3.py`
- ✅ `test_v4.py`
- ✅ `test_v5.py`
- ✅ `test_realtime.py`
- ✅ `test_trigger.py`

### Feature Extraction
- ✅ `extract_features.py`
- ✅ `extract_features_v2.py`

### Evaluation
- ✅ `evaluate_model.py`

### API
- ✅ `app.py`

---

## 🔧 Mengubah Konfigurasi

Untuk mengubah model atau path, **edit hanya `config.py`**:

### Contoh: Mengubah Model Training

```python
# SEBELUM (hardcoded di setiap file):
# train_model.py:       MODEL_SAVE_PATH = "models/lip_reading_base.h5"
# train_v2.py:          MODEL_SAVE_PATH = "models/lip_reading_aug.h5"
# evaluate_model.py:    DEFAULT_MODEL = "models/lip_reading_aug.h5"

# SESUDAH (terpusat di config.py):
MODEL_BASE = "models/lip_reading_base_v2.h5"  # Ubah di 1 tempat!
# Semua file akan otomatis menggunakan path baru
```

### Contoh: Mengubah Hyperparameter

```python
# Di config.py
EPOCHS = 150  # Ubah dari 100 ke 150
BATCH_SIZE = 32  # Ubah dari 16 ke 32

# Semua training script akan otomatis pakai nilai baru
```

---

## 💡 Best Practices

1. **Jangan hardcode** path atau konstanta di dalam script
2. **Selalu import dari `config.py`** untuk consistency
3. **Gunakan `get_model_path()`** jika perlu model selection dinamis
4. **Call `ensure_directories_exist()`** di awal main script
5. **Update `config.py`** jika ada perubahan struktur project

---

## 📊 Contoh Lengkap

### Training Script
```python
from config import (
    DATA_AUG_PATH, LABEL_AUG_PATH, LABEL_MAP_PATH, MODEL_AUG,
    EPOCHS, BATCH_SIZE, RANDOM_STATE, ensure_directories_exist
)

ensure_directories_exist()

X = np.load(DATA_AUG_PATH)
y = np.load(LABEL_AUG_PATH)

# Train model...
model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE)
model.save(MODEL_AUG)
```

### Testing Script
```python
from config import MODEL_DEFAULT, LABEL_MAP_PATH

model = tf.keras.models.load_model(MODEL_DEFAULT)

with open(LABEL_MAP_PATH) as f:
    label_map = json.load(f)

# Test model...
```

---

**✨ Kesimpulan:** Config.py adalah single source of truth untuk semua konfigurasi project. Update di satu tempat = berubah di mana-mana!
