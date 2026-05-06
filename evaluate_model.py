import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# --- KONFIGURASI ---
DATA_PATH = "processed_data/X_data.npy"
LABEL_PATH = "processed_data/y_data.npy"
MAP_PATH = "processed_data/label_map.json"
MODEL_PATH = "models/lip_reading_model.h5"
REPORT_DIR = "reports"

def evaluate():
    # 1. Cek folder laporan
    os.makedirs(REPORT_DIR, exist_ok=True)

    # 2. Load Data & Model
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model {MODEL_PATH} tidak ditemukan!")
        return

    X = np.load(DATA_PATH)
    y = np.load(LABEL_PATH)
    
    with open(MAP_PATH, 'r') as f:
        label_map = json.load(f)
    
    # Urutkan nama kelas sesuai index
    target_names = [word for word, idx in sorted(label_map.items(), key=lambda item: item[1])]
    
    print("⌛ Loading Model...")
    model = tf.keras.models.load_model(MODEL_PATH)

    # 3. Split Data (Samakan dengan train_model.py agar adil)
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 4. Prediksi
    print("🚀 Melakukan Prediksi...")
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # 5. Classification Report
    report = classification_report(y_test, y_pred, target_names=target_names)
    print("\n=== CLASSIFICATION REPORT ===")
    print(report)

    # Simpan report ke file teks
    with open(os.path.join(REPORT_DIR, "classification_report.txt"), "w") as f:
        f.write(report)

    # 6. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=target_names, yticklabels=target_names)
    plt.title('Confusion Matrix - Lip Reading Model')
    plt.ylabel('Actual Word')
    plt.xlabel('Predicted Word')
    
    # Simpan visualisasi
    cm_path = os.path.join(REPORT_DIR, "confusion_matrix.png")
    plt.savefig(cm_path)
    print(f"\n✅ Confusion Matrix disimpan di: {cm_path}")
    
    plt.show()

if __name__ == "__main__":
    evaluate()
