# import pandas as pd
# import numpy as np
# import sklearn
# import imblearn
# import matplotlib
# import joblib
# import tldextract
# import fastapi
# import uvicorn
# import streamlit
# import warnings
# warnings.filterwarnings("ignore", category=UserWarning)

# print("hghhb")
# import tensorflow as tf
# print(tf.__version__)
# import pandas as pd

# # Load the dataset
# df = pd.read_csv("data/raw/phishing.csv", encoding="latin-1")


# print("Dataset loaded successfully!")
# print("Shape:", df.shape)   # rows, columns
# print("\nFirst 5 rows:\n")
# print(df.head())

# print("\nColumn names:", df.columns.tolist())
# print("\nLabel value counts:\n")
# print(df['label'].value_counts())
from src.features import features_from_url, batch_extract

# Test with a single URL
url = "http://192.168.0.1/login?user=abc"
features = features_from_url(url)
print("Features from single URL:")
for k, v in features.items():
    print(f"{k}: {v}")

# Test with multiple URLs
urls = [
    "https://www.google.com",
    "http://paypal.com.login.verify-account.cn",
    "https://mybank.com/secure/update",
]
batch = batch_extract(urls)

print("\nFeatures from multiple URLs:")
for i, feats in enumerate(batch):
    print(f"\nURL {i+1}: {urls[i]}")
    for k, v in feats.items():
        print(f"  {k}: {v}")
