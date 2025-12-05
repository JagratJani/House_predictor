"""
evaluate.py

Load the saved model and scaler, run predictions on the test set, compute metrics,
and create predicted-vs-actual plots. Use this after train.py completes.
"""

import os
import argparse
import numpy as np
import joblib
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from train import RegressionNet  # assumes same project structure

def load_npz(npz_path):
    data = np.load(npz_path)
    return data["X_test"], data["y_test"]

def main(args):
    device = torch.device("cuda" if (torch.cuda.is_available() and not args.force_cpu) else "cpu")
    print("Using device:", device)

    # load test data
    X_test, y_test = load_npz(args.npz)

    # load model
    model = RegressionNet(X_test.shape[1]).to(device)
    state_dict = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    with torch.no_grad():
        preds = model(torch.tensor(X_test, dtype=torch.float32).to(device)).cpu().numpy().flatten()

    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    print("Test RMSE:", rmse, "MAE:", mae, "R2:", r2)

    # plot pred vs actual
    plt.figure()
    plt.scatter(y_test, preds, s=8)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)])
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Predicted vs Actual (Test)")
    plt.grid(True)
    plt.savefig(os.path.join(args.out_dir, "eval_pred_vs_actual.png"))
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", default="./data/dataset_scaled.npz", help="path to prepared npz")
    parser.add_argument("--model_path", default="./artifacts/model_best.pth")
    parser.add_argument("--out_dir", default="./artifacts")
    parser.add_argument("--force_cpu", action="store_true")
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    main(args)
