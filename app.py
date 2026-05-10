from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import tensorflow as tf
import numpy as np
import json
import os

# ==========================================
# 1. KONFIGURASI & LOAD MODEL
# ==========================================
MODEL_PATH = "models/lip_reading_aug.h5"
LABEL_MAP_PATH = "processed_data/label_map.json"

app = FastAPI(title="Lip Reading API", version="1.0")

# Izinkan CORS (Supaya bisa diakses dari Frontend JS/React/Vue)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables untuk model
model = None
target_names = []

@app.on_event("startup")
async def load_resources():
    global model, target_names
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"Model {MODEL_PATH} tidak ditemukan!")
    
    print("⌛ Loading model into memory...")
    model = tf.keras.models.load_model(MODEL_PATH)
    
    with open(LABEL_MAP_PATH, 'r') as f:
        label_map = json.load(f)
    target_names = [word for word, idx in sorted(label_map.items(), key=lambda item: item[1])]
    print("✅ Model & Label Map Loaded!")

# ==========================================
# 2. DEFINISI SCHEMA INPUT (Vectorize)
# ==========================================
class PredictionRequest(BaseModel):
    # Kita menerima list flat (vectorized)
    # Total elemen: 90 frame * 64 features = 5760
    landmarks: list[float]

# ==========================================
# 3. ENDPOINT PREDIKSI
# ==========================================
@app.post("/predict")
def predict_lip_reading(request: PredictionRequest):
    try:
        # 1. Konversi ke Numpy
        data = np.array(request.landmarks)
        
        # 2. Validasi Panjang Data
        # Harus pas 5760 (90 * 64)
        expected_size = 90 * 64
        if data.size != expected_size:
            raise HTTPException(
                status_code=400, 
                detail=f"Input size mismatch. Expected {expected_size}, got {data.size}"
            )
        
        # 3. Reshape ke format model (Batch, Time, Features)
        # Reshape ke (90, 64) lalu expand_dims jadi (1, 90, 64)
        input_data = data.reshape(1, 90, 64)
        
        # 4. Prediksi
        prediction = model.predict(input_data, verbose=0)
        predicted_idx = np.argmax(prediction)
        confidence = np.max(prediction)
        
        # 5. Return Hasil
        return {
            "success": True,
            "word": target_names[predicted_idx],
            "confidence": float(confidence),
            "all_predictions": {
                name: float(prob) for name, prob in zip(target_names, prediction[0])
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def home():
    return {"message": "Lip Reading API is running!", "model_status": "ready" if model else "loading"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
