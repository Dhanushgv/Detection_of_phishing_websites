# api.py
from __future__ import annotations
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import torch
from src.features import features_from_url
from src.dataset import prepare_sequences, CHAR2IDX
from src.cnn_model_torch import CharCNN

app = FastAPI(title="Phishing URL Detector")


rf_model = joblib.load("models/rf_model.joblib")

# Torch CNN
device = torch.device("mps" if torch.backends.mps.is_available()
                      else "cuda" if torch.cuda.is_available() else "cpu")

vocab_size = len(CHAR2IDX) + 2
cnn_model = CharCNN(vocab_size=vocab_size)
cnn_model.load_state_dict(torch.load("models/cnn_best_torch.pt", map_location=device))
cnn_model.to(device)
cnn_model.eval()


class Item(BaseModel):
    url: str

@app.post("/predict")
async def predict(item: Item):
    url = item.url

    
    rf_feat = pd.DataFrame([features_from_url(url)])
    rf_p = float(rf_model.predict_proba(rf_feat)[:, 1][0])

   
    X = prepare_sequences([url], max_len=200)
    X_tensor = torch.tensor(X, dtype=torch.long).to(device)
    with torch.no_grad():
        cnn_p = float(cnn_model(X_tensor).cpu().numpy().reshape(-1)[0])

  
    prob = (rf_p + cnn_p) / 2.0
    label = int(prob >= 0.5)

    return {
        "url": url,
        "prob_phishing": round(prob, 4),
        "label": label,
        "rf": round(rf_p, 4),
        "cnn": round(cnn_p, 4),
        "device": str(device),
        "reachable": True
    }
