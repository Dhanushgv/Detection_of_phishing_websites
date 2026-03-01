# src/predict_torch.py
from __future__ import annotations
import warnings
from urllib3.exceptions import NotOpenSSLWarning
warnings.filterwarnings("ignore", category=NotOpenSSLWarning)

import argparse, joblib
import torch
import pandas as pd

from src.cnn_model_torch import CharCNN
from src.dataset import prepare_sequences, CHAR2IDX
from src.features import features_from_url

def load_models(device):
    """Load Random Forest and Torch CNN models."""
    vocab_size = len(CHAR2IDX) + 2  # PAD + UNK

    # Load Random Forest
    rf = joblib.load("models/rf_model.joblib")

    # Load Torch CNN
    model = CharCNN(vocab_size=vocab_size).to(device)
    model.load_state_dict(torch.load("models/cnn_best_torch.pt", map_location=device))
    model.eval()

    return rf, model, device

def predict_url(url: str, rf, cnn, device, max_len: int = 200) -> dict:
    # --- Extract features from URL ---
    feats = features_from_url(url)
    rf_feat = pd.DataFrame([feats])
    reachable = bool(feats.get("reachable", 0))  # ✅ we’ll show this

    # --- RF prediction ---
    rf_p = float(rf.predict_proba(rf_feat)[:, 1][0]) * 100.0  # in %

    # --- CNN prediction ---
    X = prepare_sequences([url], max_len=max_len)
    X = torch.tensor(X, dtype=torch.long, device=device)
    with torch.no_grad():
        logits = cnn(X)
        prob = torch.sigmoid(logits).cpu().item() * 100.0
        cnn_p = float(prob)

    # --- Weighted ensemble ---
    final_prob = (0.7 * rf_p + 0.3 * cnn_p)
    label = int(final_prob >= 50.0)

    return {
        "url": url,
        "prob_phishing": round(final_prob, 2),
        "label": label,
        "rf": round(rf_p, 2),
        "cnn": round(cnn_p, 2),
        "reachable": reachable,
    }

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("url")
    p.add_argument("--max_len", type=int, default=200)
    args = p.parse_args()

    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    print(f"⚡ Using device: {device}")

    rf, cnn, device = load_models(device)
    res = predict_url(args.url, rf, cnn, device, args.max_len)

    # --- Print results ---
    reach_msg = "✅ URL reachable" if res['reachable'] else "❌ URL not reachable"
    print(reach_msg)
    print(
        f"{res['url']} -> phishing prob={res['prob_phishing']}% "
        f"| rf={res['rf']}% | cnn={res['cnn']}% | label={res['label']}"
    )
