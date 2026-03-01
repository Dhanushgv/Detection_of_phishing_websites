# src/train_lstm.py
from __future__ import annotations
import argparse, os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from src.dataset import prepare_sequences, CHAR2IDX
from src.lstm_model_torch import CharLSTM

class URLDataset(Dataset):
    def __init__(self, urls: list[str], labels: list[int], max_len: int = 200):
        self.X = prepare_sequences(urls, max_len=max_len)
        self.y = np.array(labels, dtype=np.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.long), torch.tensor(self.y[idx], dtype=torch.float32)


def train_lstm_model(
    csv_path: str,
    model_out: str = "models/lstm_best_torch.pt",
    max_len: int = 200,
    batch_size: int = 64,
    epochs: int = 8,
    lr: float = 1e-3,
    patience: int = 3,
):
    df = pd.read_csv(csv_path)
    label_col = next((c for c in ["label", "is_phishing", "phishing"] if c in df.columns), None)
    if label_col is None:
        raise ValueError("No label column found in dataset")

    urls = df["url"].astype(str).tolist()
    y = df[label_col].astype(int).tolist()

    X_train, X_val, y_train, y_val = train_test_split(urls, y, test_size=0.2, stratify=y, random_state=42)

    train_ds = URLDataset(X_train, y_train, max_len=max_len)
    val_ds = URLDataset(X_val, y_val, max_len=max_len)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"⚡ Training on: {device}")

    vocab_size = len(CHAR2IDX) + 2
    model = CharLSTM(vocab_size=vocab_size).to(device)

    class_weights = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
    class_weights = torch.tensor(class_weights, dtype=torch.float32, device=device)
    print("Class weights:", class_weights)

    criterion = nn.BCELoss(reduction="none")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for Xb, yb in loop:
            Xb, yb = Xb.to(device), yb.to(device)

            optimizer.zero_grad()
            preds = model(Xb)
            sample_weights = torch.tensor([class_weights[int(label.item())] for label in yb], device=device)
            loss = criterion(preds, yb)
            loss = (loss * sample_weights).mean()

            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(yb)
            loop.set_postfix(loss=loss.item())

        train_loss = total_loss / len(train_ds)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                preds = model(Xb)
                loss = nn.BCELoss()(preds, yb)
                val_loss += loss.item() * len(yb)
        val_loss /= len(val_ds)

        scheduler.step(val_loss)
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            os.makedirs(os.path.dirname(model_out), exist_ok=True)
            torch.save(model.state_dict(), model_out)
            print(f"✅ Saved best model at epoch {epoch+1}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("⏹ Early stopping triggered.")
                break

    print("🎉 Training complete.")
