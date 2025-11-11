# -*- coding: utf-8 -*-
"""
inference script for CatBoost CVD model
- Reproduces full feature engineering (32 features) consistent with training
- Loads model and scaler robustly (timestamped or plain filenames)
- Applies scaling to numeric features only
- Uses thresholds from config: best-config (0.48) and optional global (0.50)
"""

import os
import json
import glob
import pickle
import argparse
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = BASE_DIR  # adjust if model sits elsewhere
CONFIG_PATHS = [
    os.path.join(MODEL_DIR, "model_config.json"),               # models/model_config.json
    os.path.join(BASE_DIR, "model_config.json"),                # fallback
]

# -----------------------------------------------------------------------------
# Load config (feature names, categorical indices, optimal threshold)
# -----------------------------------------------------------------------------
def load_config():
    for p in CONFIG_PATHS:
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f), p
    raise FileNotFoundError("model_config.json not found in expected locations")

config, config_path = load_config()

FEATURE_NAMES = config.get("feature_names", [])
CAT_IDXS = config.get("categorical_features", [])
BEST_CONFIG_THRESHOLD = float(config.get("optimal_threshold", 0.48))  # from training config
GLOBAL_DEFAULT_THRESHOLD = 0.50  # from mega-analysis aggregation (operational default)

# -----------------------------------------------------------------------------
# Robust model/scaler loaders (support timestamped or plain filenames)
# -----------------------------------------------------------------------------
def find_file(patterns):
    for pat in patterns:
        candidates = glob.glob(pat)
        if candidates:
            # pick the most recent
            candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
            return candidates[0]
    return None

def load_model_and_scaler():
    # Model: try timestamped then plain name
    model_path = find_file([
        os.path.join(MODEL_DIR, "catboost_model_*.cbm"),
        os.path.join(MODEL_DIR, "catboost_model.cbm"),
    ])
    if model_path is None:
        raise FileNotFoundError("CatBoost model (.cbm) not found")
    model = CatBoostClassifier()
    model.load_model(model_path)

    # Scaler: try timestamped then plain name
    scaler_path = find_file([
        os.path.join(MODEL_DIR, "scaler_*.pkl"),
        os.path.join(MODEL_DIR, "scaler.pkl"),
    ])
    if scaler_path is None:
        raise FileNotFoundError("Scaler (.pkl) not found")
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    return model, scaler, model_path, scaler_path

# -----------------------------------------------------------------------------
# Feature engineering (must mirror training exactly)
# -----------------------------------------------------------------------------
def engineer_features_full(df):
    df = df.copy()
    eps = 1e-6

    # 1) BP features
    if {"ap_hi", "ap_lo"}.issubset(df.columns):
        df["bp_ratio"] = df["ap_hi"] / (df["ap_lo"] + eps)
        df["map"] = (df["ap_hi"] + 2 * df["ap_lo"]) / 3.0
        df["pulse_pressure"] = df["ap_hi"] - df["ap_lo"]

    # 2) BMI
    if {"height", "weight"}.issubset(df.columns):
        df["BMI"] = df["weight"] / ((df["height"] / 100.0) ** 2)

    # 3) Age transforms
    if "age" in df.columns:
        df["age_years"] = df["age"] / 365.25
        df["age_squared"] = df["age_years"] ** 2
        # if you used age_group during training, ensure identical bins/labels; not used in final 32 list

    # 4) Age × risk interactions
    if {"pulse_pressure", "age_years"}.issubset(df.columns):
        df["pp_age_ratio"] = df["pulse_pressure"] / (df["age_years"] + eps)
        df["pp_age_product"] = df["pulse_pressure"] * df["age_years"]
    if {"BMI", "age_years"}.issubset(df.columns):
        df["bmi_age_product"] = df["BMI"] * df["age_years"]
        df["bmi_age_ratio"] = df["BMI"] / (df["age_years"] + eps)
        df["bmi_squared"] = df["BMI"] ** 2
    if {"ap_hi", "age_years"}.issubset(df.columns):
        df["sbp_age_ratio"] = df["ap_hi"] / (df["age_years"] + eps)
        df["sbp_age_product"] = df["ap_hi"] * df["age_years"]

    # 5) BMI × BP
    if {"ap_hi", "BMI"}.issubset(df.columns):
        df["sbp_bmi_product"] = df["ap_hi"] * df["BMI"]
        df["sbp_bmi_ratio"] = df["ap_hi"] / (df["BMI"] + eps)
    if {"pulse_pressure", "BMI"}.issubset(df.columns):
        df["pp_bmi_product"] = df["pulse_pressure"] * df["BMI"]
        df["pp_bmi_ratio"] = df["pulse_pressure"] / (df["BMI"] + eps)

    # 6) Logs
    if "BMI" in df.columns:
        df["log_bmi"] = np.log1p(df["BMI"])
    if "pulse_pressure" in df.columns:
        df["log_pp"] = np.log1p(df["pulse_pressure"])
    if "ap_hi" in df.columns:
        df["log_sbp"] = np.log1p(df["ap_hi"])

    # 7) Lifestyle risk
    if {"cholesterol", "gluc", "smoke", "alco"}.issubset(df.columns):
        df["lifestyle_risk"] = (df["cholesterol"] / 3.0 + df["gluc"] / 3.0 + df["smoke"] + df["alco"])

    # 8) CV risk interactions
    if {"ap_hi", "cholesterol"}.issubset(df.columns):
        df["bp_chol_interaction"] = df["ap_hi"] * df["cholesterol"]
    if {"BMI", "cholesterol", "smoke"}.issubset(df.columns):
        df["metabolic_risk"] = (df["BMI"] / 30.0) * df["cholesterol"] * (1.0 + df["smoke"])

    return df

# -----------------------------------------------------------------------------
# Prepare single sample for inference
# -----------------------------------------------------------------------------
def prepare_sample(raw_dict, scaler):
    # Build DataFrame with original raw keys
    df_raw = pd.DataFrame([raw_dict])

    # Engineer features to match training
    df_feat = engineer_features_full(df_raw)

    # Ensure columns are in the exact order expected by the model
    missing = [c for c in FEATURE_NAMES if c not in df_feat.columns]
    if missing:
        raise ValueError(f"Missing engineered features: {missing}")

    X = df_feat[FEATURE_NAMES].copy()

    # Identify categorical vs numeric by indices from config
    cat_cols = [FEATURE_NAMES[i] for i in CAT_IDXS]
    num_cols = [c for c in FEATURE_NAMES if c not in cat_cols]

    # Scale numeric only
    X[num_cols] = scaler.transform(X[num_cols])

    # Return X and categorical feature indices
    return X, CAT_IDXS

# -----------------------------------------------------------------------------
# Predict function
# -----------------------------------------------------------------------------
def predict_patient(model, scaler, patient_dict, threshold_choice="global"):
    X, cat_idxs = prepare_sample(patient_dict, scaler)
    pool = Pool(X, cat_features=cat_idxs)

    prob = float(model.predict_proba(pool)[:, 1][0])

    if threshold_choice == "best":
        thr = BEST_CONFIG_THRESHOLD
    else:
        thr = GLOBAL_DEFAULT_THRESHOLD

    pred = int(prob >= thr)
    return prob, pred, thr

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Predict CVD risk for a single patient")
    parser.add_argument("--threshold", choices=["global", "best"], default="global",
                        help="Use 'global' (0.50) or 'best' (from config, e.g., 0.48)")
    args = parser.parse_args()

    model, scaler, model_path, scaler_path = load_model_and_scaler()

    # Example patient (age in days)
    new_patient = {
        "age": 18393,
        "gender": 2,       # 1=male, 2=female
        "height": 168,
        "weight": 62,
        "ap_hi": 110,
        "ap_lo": 80,
        "cholesterol": 1,  # 1/2/3
        "gluc": 1,         # 1/2/3
        "smoke": 0,
        "alco": 0,
        "active": 1,
    }

    prob, pred, thr = predict_patient(model, scaler, new_patient, threshold_choice=args.threshold)

    print(f"Model: {os.path.basename(model_path)} | Scaler: {os.path.basename(scaler_path)}")
    print(f"Threshold policy: {args.threshold} ({thr:.2f})")
    print(f"CVD Risk Probability: {prob*100:.2f}%")
    print(f"Prediction: {'CVD' if pred == 1 else 'No CVD'}")

if __name__ == "__main__":
    main()

