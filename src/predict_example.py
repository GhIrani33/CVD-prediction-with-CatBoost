# -*- coding: utf-8 -*-
"""
Example: How to use the trained CatBoost model for prediction
"""

import pickle
import pandas as pd
from catboost import CatBoostClassifier

# Load model
model = CatBoostClassifier()
model.load_model("catboost_model.cbm")

# Load scaler
with open("scaler_20251110_231143.pkl", "rb") as f:
    scaler = pickle.load(f)

# Example: Predict for new patient
new_patient = {
    'age': 18393,  # days
    'gender': 2,   # 1=male, 2=female
    'height': 168,
    'weight': 62,
    'ap_hi': 110,
    'ap_lo': 80,
    'cholesterol': 1,  # 1=normal, 2=above normal, 3=well above normal
    'gluc': 1,
    'smoke': 0,
    'alco': 0,
    'active': 1
}

# Preprocess (same feature engineering as training)
# ... (add feature engineering code here)

# Scale numeric features
# numeric_cols = [...]  # list of numeric column names
# new_patient_scaled = scaler.transform(new_patient[numeric_cols])

# Predict probability
prob = model.predict_proba(new_patient_df)[:, 1][0]

# Apply optimal threshold
threshold = 0.48
prediction = 1 if prob >= threshold else 0

print(f"CVD Risk Probability: {prob*100:.2f}%")
print(f"Prediction: {'CVD' if prediction == 1 else 'No CVD'}")
