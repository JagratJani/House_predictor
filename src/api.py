# src/api.py

import os
from typing import List

import numpy as np
import torch
import torch.nn as nn
from fastapi import FastAPI
from pydantic import BaseModel
import joblib

# -------------------
# 1. Model definition (same as in train.py)
# -------------------

class RegressionNet(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# -------------------
# 2. Pydantic schema for request body
# -------------------
# California Housing features (in order):
# ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']

class HouseFeatures(BaseModel):
    MedInc: float       # median income in block group
    HouseAge: float     # median house age in block group
    AveRooms: float     # average number of rooms
    AveBedrms: float    # average number of bedrooms
    Population: float   # block group population
    AveOccup: float     # average house occupancy
    Latitude: float     # block group latitude
    Longitude: float    # block group longitude


# -------------------
# 3. FastAPI app
# -------------------

app = FastAPI(
    title="House Price Prediction API",
    description="Predict California house prices using a PyTorch regression model.",
    version="1.0.0",
)

# Paths to artifacts
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCALER_PATH = os.path.join(BASE_DIR, "data", "scaler.joblib")
MODEL_PATH = os.path.join(BASE_DIR, "artifacts", "model_best.pth")

# Global objects loaded at startup
model: RegressionNet | None = None
scaler = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@app.on_event("startup")
def load_artifacts():
    """
    Load scaler and model into memory when the API starts.
    """
    global model, scaler

    # 1) Load scaler
    if not os.path.exists(SCALER_PATH):
        raise RuntimeError(f"Scaler not found at {SCALER_PATH}. Train first to generate it.")
    scaler = joblib.load(SCALER_PATH)

    # 2) Initialize and load model
    input_dim = 8  # number of features in California housing dataset
    model = RegressionNet(input_dim)
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"Model file not found at {MODEL_PATH}. Train first to generate it.")
    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    print("✅ Loaded scaler and model successfully.")


@app.get("/")
def read_root():
    return {
        "message": "House Price Prediction API is running.",
        "docs": "Visit /docs for interactive Swagger UI.",
    }


@app.post("/predict")
def predict_price(features: HouseFeatures):
    """
    Predict house price for a single data point.

    Returns:
    - predicted_value (in 100k USD units, like the original dataset)
    - predicted_value_dollars (approximate in USD)
    """

    if model is None or scaler is None:
        raise RuntimeError("Model or scaler not loaded. Check startup logs.")

    # 1. Convert input to numpy array with correct order & shape (1, 8)
    x = np.array([[
        features.MedInc,
        features.HouseAge,
        features.AveRooms,
        features.AveBedrms,
        features.Population,
        features.AveOccup,
        features.Latitude,
        features.Longitude,
    ]], dtype=np.float32)

    # 2. Scale using the same scaler used in training
    x_scaled = scaler.transform(x)

    # 3. Convert to tensor and move to device
    x_tensor = torch.tensor(x_scaled, dtype=torch.float32).to(device)

    # 4. Predict
    with torch.no_grad():
        pred = model(x_tensor).cpu().numpy().flatten()[0]

    # Original target is in units of 100k dollars
    pred_100k = float(pred)
    pred_dollars = float(pred * 100000.0)

    return {
        "predicted_value_100k": pred_100k,
        "predicted_value_dollars": pred_dollars,
    }
