from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
from typing import Optional, Dict, Any
import pandas as pd
from pathlib import Path

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "model.joblib"

app = FastAPI(title="House Price Predictor API")


class Features(BaseModel):
    # Accept arbitrary features; we'll validate in code
    data: Dict[str, Any]


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Train first using train_and_save.py")
    return joblib.load(MODEL_PATH)


model = None


def startup_event():
    global model
    model = load_model()


def health():
    return {"status": "ok", "model": str(MODEL_PATH)}


def predict(payload: Features):
    global model
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    data = payload.data
    try:
        df = pd.DataFrame([data])
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid input data: {e}")

    try:
        preds = model.predict(df)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error during prediction: {e}")

    return {"prediction": float(preds[0])}
