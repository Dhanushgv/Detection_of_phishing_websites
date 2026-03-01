# # src/train_cnn_torch.py
# from __future__ import annotations
# import argparse, os
# import numpy as np
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.utils.class_weight import compute_class_weight
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader, TensorDataset
# from tqdm import tqdm

# from src.dataset import prepare_sequences, CHAR2IDX
# from src.cnn_model_torch import CharCNN   # ✅ Torch CNN

# def load_data(csv_path: str):
#     df = pd.read_csv(csv_path)
#     label_col = next((c for c in ["label","Label","result","Result","is_phishing","phishing"] if c in df.columns), None)
#     if label_col is None:
#         raise ValueError("No label column found")
#     if "url" not in df.columns:
#         raise ValueError("Dataset must have a 'url' column")
#     urls = df["url"].astype(str).tolist()
#     y = df[label_col].astype(int).values
#     return urls, y


# def main(args):
#     device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
#     print(f"⚡ Using device: {device}")

#     urls, y = load_data(args.data)
#     X = prepare_sequences(urls, max_len=args.max_len)

#     X_train, X_val, y_train, y_val = train_test_split(
#         X, y, test_size=0.2, stratify=y, random_state=42
#     )

#     # Convert to tensors
#     X_train, y_train = torch.tensor(X_train, dtype=torch.long), torch.tensor(y_train, dtype=torch.float32)
#     X_val, y_val = torch.tensor(X_val, dtype=torch.long), torch.tensor(y_val, dtype=torch.float32)

#     train_ds = TensorDataset(X_train, y_train)
#     val_ds = TensorDataset(X_val, y_val)
#     train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
#     val_loader = DataLoader(val_ds, batch_size=args.batch)

#     # Model
#     model = CharCNN(vocab_size=max(CHAR2IDX.values())+1, max_len=args.max_len).to(device)
#     optimizer = optim.Adam(model.parameters(), lr=1e-3)
#     criterion = nn.BCELoss()

#     best_val_acc = 0.0
#     os.makedirs("models", exist_ok=True)

#     for epoch in range(1, args.epochs+1):
#         model.train()
#         pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
#         for xb, yb in pbar:
#             xb, yb = xb.to(device), yb.to(device)
#             optimizer.zero_grad()
#             preds = model(xb).squeeze()
#             loss = criterion(preds, yb)
#             loss.backward()
#             optimizer.step()
#             pbar.set_postfix(loss=loss.item())

#         # Validation
#         model.eval()
#         correct, total = 0, 0
#         with torch.no_grad():
#             for xb, yb in val_loader:
#                 xb, yb = xb.to(device), yb.to(device)
#                 preds = (model(xb).squeeze() >= 0.5).float()
#                 correct += (preds == yb).sum().item()
#                 total += yb.size(0)
#         val_acc = correct / total
#         print(f"✅ Epoch {epoch}: Val acc = {val_acc:.4f}")

#         # Save best model
#         if val_acc > best_val_acc:
#             best_val_acc = val_acc
#             torch.save(model.state_dict(), "models/cnn_best_torch.pt")
#             print("💾 Saved best model → models/cnn_best_torch.pt")

#     print(f"🎉 Training complete! Best val acc: {best_val_acc}")

# if __name__ == "__main__":
#     p = argparse.ArgumentParser()
#     p.add_argument("--data", default="data/raw/phishing.csv")
#     p.add_argument("--max_len", type=int, default=200)
#     p.add_argument("--epochs", type=int, default=5)
#     p.add_argument("--batch", type=int, default=256)
#     args = p.parse_args()
#     main(args)
# src/train_cnn_torch.py
from __future__ import annotations
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from src.dataset import prepare_sequences, CHAR2IDX
from src.cnn_model_torch import CharCNN

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("⚡ Using device:", device)

    # Load your phishing dataset
    df = pd.read_csv("data/raw/phishing.csv")
    urls = df["url"].astype(str).tolist()
    y = df["label"].astype(int).values

    # Prepare sequences
    X = prepare_sequences(urls, max_len=200)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.long), torch.tensor(y_train, dtype=torch.float32))
    val_ds = TensorDataset(torch.tensor(X_val, dtype=torch.long), torch.tensor(y_val, dtype=torch.float32))

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64)

    # ✅ Correct vocab size (PAD+UNK+92 chars = 94)
    vocab_size = len(CHAR2IDX) + 2
    print("Vocab size:", vocab_size)

    model = CharCNN(vocab_size=vocab_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCELoss()

    best_auc = 0
    for epoch in range(5):  # train for a few epochs
        model.train()
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            preds = model(Xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()

        # validation
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb = Xb.to(device)
                preds = model(Xb).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(yb.numpy())

        auc = roc_auc_score(all_labels, all_preds)
        print(f"Epoch {epoch+1}: val AUC={auc:.4f}")
        if auc > best_auc:
            best_auc = auc
            torch.save(model.state_dict(), "models/cnn_best_torch.pt")
            print("✅ Saved new best model")

if __name__ == "__main__":
    main()
