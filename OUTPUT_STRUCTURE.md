# 📊 Output Training Terpisah

Setiap file training sekarang **menyimpan output ke folder terpisah**. Ini memudahkan membandingkan hasil training base vs augmented.

---

## 📁 Struktur Output

```
outputs/
├── train_base/                  # Output dari train_model.py (non-augmented)
│   ├── training_history.png     # Plot accuracy & loss
│   └── training_metrics.txt     # Metrics ringkas
│
├── train_aug/                   # Output dari train_v2.py & train_v3.py (augmented)
│   ├── training_history.png     # Plot accuracy & loss
│   └── training_metrics.txt     # Metrics ringkas
│
└── train_custom/                # Output training custom lainnya (optional)
    ├── training_history.png
    └── training_metrics.txt
```

---

## 🎯 Apa isinya?

### `training_history.png`
- **Accuracy Plot**: Menunjukkan train vs validation accuracy per epoch
- **Loss Plot**: Menunjukkan train vs validation loss per epoch
- Berguna untuk lihat apakah model overfitting atau underfitting

### `training_metrics.txt`
```
==================================================
TRAINING METRICS - AUGMENTED MODEL
==================================================

Data: processed_data/X_data_aug.npy
Model: models/lip_reading_aug.h5
Train videos (original): 140
Train samples (augmented): 700
Val samples: 35
Number of classes: 17
Augmentations per video: 5
Final Train Accuracy: 0.9850
Final Val Accuracy: 0.9200
Final Train Loss: 0.0456
Final Val Loss: 0.2841
Total Epochs Trained: 85
```

---

## 🚀 Cara Menjalankan Training

### Training Base (Non-Augmented)
```bash
python train_model.py
# Output → outputs/train_base/
```

### Training Augmented (V2)
```bash
python train_v2.py
# Output → outputs/train_aug/
```

### Training Augmented dengan Callbacks (V3)
```bash
python train_v3.py
# Output → outputs/train_aug/
```

---

## 📊 Membandingkan Hasil

Sekarang Anda bisa mudah membandingkan performa:

| Metric | Base | Augmented | Selisih |
|--------|------|-----------|---------|
| Train Accuracy | 95.20% | 98.50% | +3.30% |
| Val Accuracy | 88.50% | 92.00% | +3.50% |
| Val Loss | 0.3421 | 0.2841 | -0.0580 |

👉 **Lihat file di `outputs/train_base/training_metrics.txt` dan `outputs/train_aug/training_metrics.txt`**

---

## 💡 Tips

1. **Buka `training_history.png`** untuk lihat visual training progress
2. **Banding-bandingkan `training_metrics.txt`** antar folder untuk lihat improvement
3. **Jangan delete folder** `outputs/` - itu adalah rekam jejak training Anda
4. Jika mau run training ulang, output lama akan **tertimpa**

---

## ✅ Checklist setelah Training

- [ ] Cek `training_history.png` - apakah loss turun?
- [ ] Baca `training_metrics.txt` - apa final accuracy-nya?
- [ ] Bandingkan `train_base/` vs `train_aug/` - mana yang lebih bagus?
- [ ] Model tersimpan di `models/lip_reading_*.h5` ✓
