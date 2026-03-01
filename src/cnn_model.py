# src/cnn_model.py
from __future__ import annotations
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def make_cnn(vocab_size: int, max_len: int=200, emb_dim: int=32) -> keras.Model:
    inp = keras.Input(shape=(max_len,), dtype="int32")
    x = layers.Embedding(input_dim=vocab_size, output_dim=emb_dim, mask_zero=True)(inp)
    x = layers.Conv1D(64, kernel_size=5, activation="relu")(x)
    x = layers.GlobalMaxPooling1D()(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(1, activation="sigmoid")(x)

    model = keras.Model(inp, out)
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=[keras.metrics.AUC(name="auc"), "accuracy"],
    )
    return model
