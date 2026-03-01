# src/train_rf.py
from __future__ import annotations
import argparse
import os
import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from src.features import batch_extract


# Candidate column names for label
LABEL_CANDIDATES = ["label", "Label", "result", "Result", "is_phishing", "phishing"]

# List of all numeric features we engineered in features.py
NUMERIC_FEATURES = [
    "url_length","num_digits","num_letters","num_special","num_dots","num_hyphens",
    "num_slashes","num_question","num_equal","num_at","uses_https","has_ip",
    "subdomain_count","tld_length","path_length","query_length","entropy",
    "pct_digits","pct_special","suspicious_words",
]

def load_data(csv_path: str) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(csv_path)

    # Try to find the label column
    label_col = None
    for c in LABEL_CANDIDATES:
        if c in df.columns:
            label_col = c
            break
    if label_col is None:
        raise ValueError("Could not find a label column. Expected one of: " + ", ".join(LABEL_CANDIDATES))

    # If raw URLs are present → extract features
    if "url" in df.columns:
        feats = pd.DataFrame(batch_extract(df["url"].astype(str).tolist()))
        X = feats
    else:
        # assume features already present
        missing = [f for f in NUMERIC_FEATURES if f not in df.columns]
        if missing:
            raise ValueError(f"Missing expected feature columns: {missing}")
        X = df[NUMERIC_FEATURES].copy()

    y = df[label_col].astype(int)
    return X, y

def build_pipeline():
    pre = ColumnTransformer([
        ("num", StandardScaler(with_mean=False), NUMERIC_FEATURES),
    ], remainder='drop')

    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=1,
        max_features='sqrt',
        class_weight='balanced',
        n_jobs=-1,
        random_state=42,
    )

    # Pipeline: scaling → oversampling → Random Forest
    pipe = Pipeline([
        ("pre", pre),
        ("ros", RandomOverSampler(random_state=42)),
        ("rf", rf),
    ])
    return pipe

def main(args):
    X, y = load_data(args.data)

    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    pipe = build_pipeline()

    # Grid search hyperparameters
    param_grid = {
        "rf__n_estimators": [200, 400],
        "rf__max_depth": [None, 20, 40],
        "rf__min_samples_leaf": [1, 2],
        "rf__max_features": ['sqrt', 0.5],
    }

    gs = GridSearchCV(
        pipe,
        param_grid,
        scoring='roc_auc',
        n_jobs=-1,
        cv=3,
        verbose=1,
    )
    gs.fit(X_train, y_train)

    best = gs.best_estimator_
    y_pred = best.predict(X_test)
    y_proba = best.predict_proba(X_test)[:, 1]

    print("Best params:", gs.best_params_)
    print("ROC AUC:", roc_auc_score(y_test, y_proba))
    print(classification_report(y_test, y_pred))

    # Save model + feature order
    os.makedirs("models", exist_ok=True)
    joblib.dump(best, "models/rf_model.joblib")
    with open("models/rf_features.json", "w") as f:
        json.dump({"feature_order": NUMERIC_FEATURES}, f)

    print("✅ Model saved to: models/rf_model.joblib")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/raw/phishing.csv")
    args = p.parse_args()
    main(args)
