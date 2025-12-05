"""
train.py

Train a PyTorch regression model on the prepared dataset.
Saves:
 - best model state_dict (model_best.pth)
 - last model (model_final.pth)
 - training history JSON and training/val loss plot
"""

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import joblib

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import mean_squared_error

from data import load_and_prepare  # our helper from data.py


class RegressionNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x)


def create_loaders(X_train, y_train, X_val, y_val, batch_size=64):
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)

    train_ds = TensorDataset(X_train_t, y_train_t)
    val_ds = TensorDataset(X_val_t, y_val_t)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


def train_loop(model, train_loader, val_loader, device, epochs, lr, out_dir, patience=15):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    best_val = float("inf")
    counter = 0
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(1, epochs + 1):
        # ----- train -----
        model.train()
        running = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()

            running += loss.item() * xb.size(0)

        train_loss = running / len(train_loader.dataset)
        history["train_loss"].append(train_loss)

        # ----- validation -----
        model.eval()
        running_val = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                preds = model(xb)
                loss = criterion(preds, yb)
                running_val += loss.item() * xb.size(0)

        val_loss = running_val / len(val_loader.dataset)
        history["val_loss"].append(val_loss)

        print(f"Epoch {epoch:03d}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")

        # ----- early stopping / checkpoint -----
        if val_loss < best_val - 1e-8:
            best_val = val_loss
            counter = 0
            torch.save(model.state_dict(), os.path.join(out_dir, "model_best.pth"))
        else:
            counter += 1

        if counter >= patience:
            print("Early stopping triggered.")
            break

    # save final model
    torch.save(model.state_dict(), os.path.join(out_dir, "model_final.pth"))

    # save history
    with open(os.path.join(out_dir, "train_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    # plot losses
    plt.figure()
    plt.plot(history["train_loss"], label="train")
    plt.plot(history["val_loss"], label="val")
    plt.xlabel("Epoch")
    plt.ylabel("MSE loss")
    plt.legend()
    plt.grid(True)
    plt.title("Train vs Val loss")
    plt.savefig(os.path.join(out_dir, "loss_plot.png"))
    plt.close()

    return history


def main(args):
    os.makedirs(args.out_dir, exist_ok=True)

    # load or prepare data
    if args.npz and os.path.exists(args.npz):
        data = np.load(args.npz)
        X_train = data["X_train"]
        y_train = data["y_train"]
        X_val = data["X_val"]
        y_val = data["y_val"]
        X_test = data["X_test"]
        y_test = data["y_test"]
    else:
        info = load_and_prepare(out_dir=args.data_dir, save_npz=True)
        npz_path = info["npz_path"]
        data = np.load(npz_path)
        X_train = data["X_train"]
        y_train = data["y_train"]
        X_val = data["X_val"]
        y_val = data["y_val"]
        X_test = data["X_test"]
        y_test = data["y_test"]

    # choose device
    device = torch.device("cuda" if (torch.cuda.is_available() and not args.force_cpu) else "cpu")
    print("Using device:", device)

    # build loaders & model
    train_loader, val_loader = create_loaders(
        X_train, y_train, X_val, y_val, batch_size=args.batch_size
    )

    model = RegressionNet(X_train.shape[1]).to(device)

    # train
    history = train_loop(
        model,
        train_loader,
        val_loader,
        device,
        epochs=args.epochs,
        lr=args.lr,
        out_dir=args.out_dir,
        patience=args.patience,
    )

    # quick test evaluation with best model
    model.load_state_dict(
        torch.load(os.path.join(args.out_dir, "model_best.pth"), map_location=device)
    )
    model.eval()
    with torch.no_grad():
        X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
        preds = model(X_test_t).cpu().numpy().flatten()

    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = float(np.mean(np.abs(y_test - preds)))
    r2 = float(1 - np.sum((y_test - preds) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2))

    print(f"Test RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")

    metrics = {"rmse": float(rmse), "mae": mae, "r2": r2}
    with open(os.path.join(args.out_dir, "test_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="./data", help="where to cache/prep data")
    parser.add_argument("--npz", default=None, help="optional pre-saved npz path")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out_dir", default="./artifacts")
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--force_cpu", action="store_true")
    args = parser.parse_args()
    main(args)
