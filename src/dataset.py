
from __future__ import annotations
import json
import numpy as np

# ✅ Character vocabulary (94 total: PAD=0, UNK=1, plus 92 printable chars)
CHARS = (
    "abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "0123456789"
    "-_.@/:?=&%#~+!*'(),;[]{}|\\\"`^$"
)

PAD = 0
UNK = 1
CHAR2IDX = {c: i + 2 for i, c in enumerate(CHARS)}  # shift by 2

def encode_url(url: str, max_len: int = 200) -> list[int]:
    seq = [CHAR2IDX.get(c, UNK) for c in url]
    if len(seq) >= max_len:
        return seq[:max_len]
    return seq + [PAD] * (max_len - len(seq))

def prepare_sequences(urls: list[str], max_len: int = 200) -> np.ndarray:
    return np.array([encode_url(u, max_len) for u in urls], dtype=np.int64)

def save_vocab(path: str):
    with open(path, "w") as f:
        json.dump({"PAD": PAD, "UNK": UNK, "char2idx": CHAR2IDX}, f)

if __name__ == "__main__":
    print("CHARS length:", len(CHARS))
    print("Unique chars:", len(set(CHARS)))
    print("Vocab size (with PAD+UNK):", len(CHAR2IDX) + 2)
    print("CHARS:", CHARS)
