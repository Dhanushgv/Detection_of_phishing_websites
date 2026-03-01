# import pandas as pd

# RAW_PATH = "data/raw/phishing.csv"
# CLEAN_PATH = "data/raw/phishing_clean.csv"

# try:
#     # Try loading with tolerant parsing
#     df = pd.read_csv(RAW_PATH, encoding="latin-1", on_bad_lines="skip")
# except Exception as e:
#     print("❌ Error reading CSV:", e)
#     exit()

# print("Original columns:", df.columns.tolist())
# print("Original shape:", df.shape)

# # --- Step 1: Try to find URL column ---
# url_col_candidates = [c for c in df.columns if "url" in c.lower()]
# if url_col_candidates:
#     url_col = url_col_candidates[0]
# else:
#     # fallback: take first column
#     url_col = df.columns[0]

# # --- Step 2: Try to find label column ---
# label_col_candidates = [c for c in df.columns if "label" in c.lower() or "status" in c.lower() or "result" in c.lower()]
# if label_col_candidates:
#     label_col = label_col_candidates[0]
# else:
#     # fallback: take last column
#     label_col = df.columns[-1]

# print(f"Using URL column: {url_col}, Label column: {label_col}")

# # --- Step 3: Keep only url + label ---
# df = df[[url_col, label_col]].rename(columns={url_col: "url", label_col: "label"})

# # --- Step 4: Normalize label values ---
# # Map common variants to 0/1
# df["label"] = df["label"].astype(str).str.lower().map({
#     "bad": 1, "phishing": 1, "malicious": 1, "1": 1, "-1": 1,
#     "good": 0, "legit": 0, "benign": 0, "0": 0, "safe": 0
# })

# # Drop rows with unmapped labels
# df = df.dropna(subset=["label"]).astype({"label": "int"})

# # --- Step 5: Save cleaned version ---
# df.to_csv(CLEAN_PATH, index=False)

# print("✅ Cleaned dataset saved to:", CLEAN_PATH)
# print("Shape after cleaning:", df.shape)
# print(df.head())
# print("\nLabel distribution:\n", df["label"].value_counts())
import pandas as pd

df = pd.read_csv("data/raw/phishing_clean.csv")

print("✅ Cleaned dataset loaded!")
print("Shape:", df.shape)
print("\nFirst 5 rows:\n", df.head())
print("\nLabel distribution:\n", df["label"].value_counts())
