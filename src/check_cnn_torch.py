# src/check_cnn_torch.py
from __future__ import annotations
import torch
import argparse

from src.cnn_model_torch import CharCNN
from src.dataset import prepare_sequences, CHAR2IDX

def load_cnn(model_path="models/cnn_best_torch.pt", max_len=200):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"⚡ Using device: {device}")

    vocab_size = len(CHAR2IDX) + 2
    print("Loading CNN with vocab size:", vocab_size)

    model = CharCNN(vocab_size=vocab_size).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model, device

def test_urls(urls, model, device, max_len=200):
    X = prepare_sequences(urls, max_len=max_len)
    X = torch.tensor(X, dtype=torch.long, device=device)
    with torch.no_grad():
        preds = model(X).cpu().numpy().reshape(-1)
    for url, p in zip(urls, preds):
        print(f"{url} -> phishing prob={p:.4f} | label={int(p >= 0.5)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/cnn_best_torch.pt")
    args = parser.parse_args()

    urls = [
        "http://secure-login.google.com.verify-account.example.ru",  # phishing
        "http://secure-login.update-account.example.com/verify",    # phishing
        "https://www.github.com",                                   # safe
        "https://www.wikipedia.org",                                # safe
    ]

    model, device = load_cnn(args.model)
    test_urls(urls, model, device)
