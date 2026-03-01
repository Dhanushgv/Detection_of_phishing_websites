# src/evaluate_torch.py
from __future__ import annotations
import argparse, joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score
import torch

from src.features import batch_extract
from src.dataset import prepare_sequences, CHAR2IDX
from src.cnn_model_torch import CharCNN


LABEL_CANDIDATES = ["label", "Label", "result", "Result", "is_phishing", "phishing"]

def load_xy(csv, limit=None):
    df = pd.read_csv(csv)
    if limit:
        df = df.sample(n=limit, random_state=42).reset_index(drop=True)
    label_col = next((c for c in LABEL_CANDIDATES if c in df.columns), None)
    if label_col is None:
        raise ValueError("No label column")
    y = df[label_col].astype(int).values
    urls = df["url"].astype(str).tolist() if "url" in df.columns else None
    return urls, y


def main(args):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"⚡ Using device: {device}")

    urls, y = load_xy(args.data, args.limit)

    # --- Random Forest ---
    rf = joblib.load("models/rf_model.joblib")
    if urls is None:
        raise ValueError("Evaluation expects a 'url' column for RF feature extraction")
    X_rf = pd.DataFrame(batch_extract(urls))
    rf_proba = rf.predict_proba(X_rf)[:,1]

    # --- Torch CNN ---
    model = CharCNN(vocab_size=max(CHAR2IDX.values())+1, max_len=args.max_len).to(device)
    model.load_state_dict(torch.load("models/cnn_best_torch.pt", map_location=device))
    model.eval()

    X_cnn = prepare_sequences(urls, max_len=args.max_len)
    X_cnn = torch.tensor(X_cnn, dtype=torch.long).to(device)

    with torch.no_grad():
        cnn_proba = model(X_cnn).squeeze().cpu().numpy()

    # --- Simple Ensemble (average RF + CNN) ---
    p = (rf_proba + cnn_proba) / 2.0
    y_pred = (p >= 0.5).astype(int)

    print("✅ Ensemble ROC AUC:", roc_auc_score(y, p))
    print(classification_report(y, y_pred))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/raw/phishing.csv")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--max_len", type=int, default=200)
    args = p.parse_args()
    main(args)
