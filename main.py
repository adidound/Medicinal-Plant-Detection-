# api/main.py

import os
import json
from io import BytesIO

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model

from src.config import MODEL_DIR, IMG_SIZE, KNOWLEDGE_BASE

app = FastAPI(title="Medicinal Plant Recognition API")

# ---------- Load model & metadata at startup ----------

MODEL_NAME = "mobilenet_v2.h5"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_NAME)
CLASS_INDICES_PATH = os.path.join(MODEL_DIR, "class_indices.json")

if not os.path.exists(MODEL_PATH):
    raise RuntimeError(f"Model file not found at {MODEL_PATH}. Train your model first.")

if not os.path.exists(CLASS_INDICES_PATH):
    raise RuntimeError(
        f"class_indices.json not found at {CLASS_INDICES_PATH}. "
        f"Run training to generate it."
    )

model = load_model(MODEL_PATH)

with open(CLASS_INDICES_PATH, "r") as f:
    class_indices = json.load(f)

# Reverse mapping: index -> class_name
idx_to_class = {v: k for k, v in class_indices.items()}

# Load medicinal uses knowledge base (you will fill this file)
plant_uses = {}
if os.path.exists(KNOWLEDGE_BASE):
    try:
        with open(KNOWLEDGE_BASE, "r") as f:
            content = f.read().strip()
            if content:
                plant_uses = json.loads(content)
                print(f"✅ Loaded plant uses for {len(plant_uses)} plants.")
            else:
                print("⚠️ plant_uses.json is empty. API will return empty info for plants.")
    except json.JSONDecodeError as e:
        print(f"⚠️ plant_uses.json is invalid JSON ({e}). API will return empty info for plants.")
else:
    print("⚠️ plant_uses.json not found, API will return empty info for plants.")


# ---------- Helper functions ----------

def preprocess_image(image: Image.Image) -> np.ndarray:
    """
    Resize and scale image same as training.
    """
    image = image.convert("RGB")
    image = image.resize(IMG_SIZE)
    img_array = np.array(image).astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def predict_plant(img_array: np.ndarray):
    preds = model.predict(img_array)[0]
    idx = int(np.argmax(preds))
    confidence = float(preds[idx])
    class_name = idx_to_class[idx]
    return class_name, confidence, preds.tolist()


# ---------- API endpoints ----------

@app.get("/")
def root():
    return {"message": "Medicinal Plant Recognition API is running."}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    content = await file.read()

    try:
        image = Image.open(BytesIO(content))
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read image file.")

    img_array = preprocess_image(image)
    class_name, confidence, prob_list = predict_plant(img_array)

    info = plant_uses.get(class_name, {})
    response = {
        "plant_id": class_name,
        "confidence": confidence,
        "probabilities": prob_list,
        "info": info
    }

    return JSONResponse(response)
