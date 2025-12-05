"""
data.py

Responsibilities:
- Load the California Housing dataset (sklearn)
- Split into train / val / test
- Fit a StandardScaler on train and transform all sets
- Save scaled numpy arrays or return them to caller

This file is designed to be importable by train.py and evaluate.py.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_prepare(out_dir="data", test_size=0.30, val_fraction_of_temp=0.5, random_state=42, save_npz=True):
    os.makedirs(out_dir, exist_ok=True)

    # 1. Load dataset
    try:
        data = fetch_california_housing(as_frame=True)
        X = data.data
        y = data.target
    except Exception as e:
        msg = ("Failed to fetch California Housing dataset. "
               "If you already have a local CSV, set the environment or modify this function to load that CSV.")
        raise RuntimeError(msg) from e

    # 2. Split: train / temp, then temp -> val/test
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=val_fraction_of_temp, random_state=random_state
    )

    # 3. Scale: fit scaler on train only
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Save scaler for inference later
    scaler_path = os.path.join(out_dir, "scaler.joblib")
    joblib.dump(scaler, scaler_path)

    # Optionally save numpy files for quick loading by train.py
    if save_npz:
        np.savez_compressed(
            os.path.join(out_dir, "dataset_scaled.npz"),
            X_train=X_train_scaled,
            y_train=y_train.to_numpy(),
            X_val=X_val_scaled,
            y_val=y_val.to_numpy(),
            X_test=X_test_scaled,
            y_test=y_test.to_numpy(),
        )

    return {
        "X_train": X_train_scaled,
        "y_train": y_train.to_numpy(),
        "X_val": X_val_scaled,
        "y_val": y_val.to_numpy(),
        "X_test": X_test_scaled,
        "y_test": y_test.to_numpy(),
        "scaler_path": scaler_path,
        "npz_path": os.path.join(out_dir, "dataset_scaled.npz") if save_npz else None,
    }

if __name__ == "__main__":
    print("Preparing data and saving to ./data")
    info = load_and_prepare(out_dir="./data")
    print("Saved scaler to:", info["scaler_path"])
    print("Saved npz to:", info["npz_path"])
