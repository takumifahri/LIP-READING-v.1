import numpy as np
import json
import os
import argparse
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from config import (
    DATA_PATH, LABEL_PATH, DATA_AUG_PATH, LABEL_AUG_PATH, LABEL_MAP_PATH,
    REPORTS_DIR, AUGMENTATIONS_PER_VIDEO, RANDOM_STATE, MODEL_DEFAULT, MODEL_AUG, MODEL_BASE
)

# Default model jika tidak ada argumen
DEFAULT_MODEL = MODEL_AUG


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluasi model lip reading dan simpan classification report + confusion matrix."
    )
    parser.add_argument(
        "model_pos",
        nargs="?",
        choices=["base", "aug"],
        help="Model yang dievaluasi. Tetap mendukung format lama: python evaluate_model.py base",
    )
    parser.add_argument(
        "-m",
        "--model",
        choices=["base", "aug"],
        help="Pilih model: base atau aug. Default: aug",
    )
    parser.add_argument(
        "--model-path",
        help="Path manual ke file .h5 jika ingin evaluasi model custom.",
    )
    parser.add_argument(
        "--data",
        choices=["base", "aug"],
        help="Dataset evaluasi untuk --model-path. Default: base untuk model custom.",
    )
    parser.add_argument(
        "--label",
        help="Label output report untuk --model-path. Default: nama file model.",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="Tampilkan pilihan model yang tersedia lalu keluar.",
    )
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        help="Izinkan TensorFlow mencoba GPU. Default evaluasi memakai CPU agar warning CUDA tidak muncul.",
    )
    return parser.parse_args()


def print_available_models():
    print("Model tersedia:")
    print(f"  base -> {MODEL_BASE} | data: {DATA_PATH}")
    print(f"  aug  -> {MODEL_AUG} | data: {DATA_AUG_PATH}")


def get_model_and_data_paths(args):
    """Pilih model dan dataset evaluasi yang sesuai."""
    if args.model_path:
        data_type = args.data or "base"
        model_label = args.label or os.path.splitext(os.path.basename(args.model_path))[0]
        if data_type == "aug":
            return model_label, args.model_path, DATA_AUG_PATH, LABEL_AUG_PATH, True
        return model_label, args.model_path, DATA_PATH, LABEL_PATH, False

    model_type = args.model or args.model_pos
    if model_type is None:
        return "aug", DEFAULT_MODEL, DATA_AUG_PATH, LABEL_AUG_PATH, True

    if model_type == "base":
        return "base", MODEL_BASE, DATA_PATH, LABEL_PATH, False
    if model_type == "aug":
        return "aug", MODEL_AUG, DATA_AUG_PATH, LABEL_AUG_PATH, True

    raise ValueError(f"Model tidak dikenal: {model_type}")


def grouped_augmented_test_split(X, y):
    """Ambil original test videos tanpa mencampur varian augmentasi video yang sama."""
    if len(X) % AUGMENTATIONS_PER_VIDEO != 0:
        raise ValueError(
            "Jumlah sampel augmented tidak habis dibagi "
            f"{AUGMENTATIONS_PER_VIDEO}. Evaluasi grouped tidak aman."
        )

    original_indices = np.arange(len(X) // AUGMENTATIONS_PER_VIDEO)
    original_labels = y[::AUGMENTATIONS_PER_VIDEO]
    _, test_original = train_test_split(
        original_indices,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=original_labels,
    )
    # Untuk metrik laporan, hitung video original saja. Varian augmentasi berguna
    # saat training, tetapi jangan ikut dihitung sebagai sampel test independen.
    test_idx = test_original * AUGMENTATIONS_PER_VIDEO
    return X[test_idx], y[test_idx]

def evaluate():
    args = parse_args()
    if args.list_models:
        print_available_models()
        return

    if not args.use_gpu:
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

    import tensorflow as tf
    import matplotlib.pyplot as plt
    import seaborn as sns

    model_label, model_path, data_path, label_path, use_grouped_aug_split = get_model_and_data_paths(args)
    if args.model is None and args.model_pos is None and args.model_path is None:
        print("ℹ️  Tidak ada model dipilih, memakai default: aug")

    print(f"🔍 Evaluating Model: {model_path}")
    print(f"🏷️  Label Evaluasi: {model_label}")
    print(f"📦 Dataset Evaluasi: {data_path}")
    
    # Gunakan variable dari config
    MAP_PATH = LABEL_MAP_PATH
    REPORT_DIR_PATH = REPORTS_DIR
    
    # 1. Cek folder laporan
    os.makedirs(REPORT_DIR_PATH, exist_ok=True)

    # 2. Load Data & Model
    if not os.path.exists(model_path):
        print(f"Error: Model {model_path} tidak ditemukan!")
        return
    if not os.path.exists(data_path) or not os.path.exists(label_path):
        print(f"Error: Dataset evaluasi {data_path} / {label_path} tidak ditemukan!")
        return

    X = np.load(data_path)
    y = np.load(label_path)
    
    with open(MAP_PATH, 'r') as f:
        label_map = json.load(f)
    
    # Urutkan nama kelas sesuai index
    target_names = [word for word, idx in sorted(label_map.items(), key=lambda item: item[1])]
    
    print("⌛ Loading Model...")
    model = tf.keras.models.load_model(model_path)

    # 3. Split Data
    if use_grouped_aug_split:
        X_test, y_test = grouped_augmented_test_split(X, y)
        print(
            "⚠️  Evaluasi augmented memakai group split dan hanya menghitung "
            "rekaman original pada holdout set."
        )
        print(
            "⚠️  Kalau model ini dilatih sebelum perbaikan group split, "
            "latih ulang dengan train_v3.py dulu agar hasilnya tidak leakage."
        )
    else:
        _, X_test, _, y_test = train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
        )
    print(f"✓ Test samples: {len(X_test)}")

    # 4. Prediksi
    print("🚀 Melakukan Prediksi...")
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # 5. Classification Report
    report = classification_report(y_test, y_pred, target_names=target_names)
    print("\n=== CLASSIFICATION REPORT ===")
    print(report)

    # Simpan report ke file teks
    report_path = os.path.join(REPORT_DIR_PATH, f"classification_report_{model_label}.txt")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"✅ Classification report disimpan di: {report_path}")

    # 6. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=target_names, yticklabels=target_names)
    plt.title('Confusion Matrix - Lip Reading Model')
    plt.ylabel('Actual Word')
    plt.xlabel('Predicted Word')
    
    # Simpan visualisasi
    cm_path = os.path.join(REPORT_DIR_PATH, f"confusion_matrix_{model_label}.png")
    plt.savefig(cm_path)
    print(f"\n✅ Confusion Matrix disimpan di: {cm_path}")
    
    plt.show()

if __name__ == "__main__":
    evaluate()
