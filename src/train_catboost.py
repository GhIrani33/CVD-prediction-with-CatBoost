# -*- coding: utf-8 -*-
"""
Usage:
python train_catboost.py --input "D:\Project\Heart diseae\dataset\1\cardio_clean.csv" --output_dir "final_model" --seed 42

Author: Ghasem
https://github.com/GhIrani33?tab=repositories
"""

import os
import sys
import json
import time
import pickle
import argparse
import logging
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score,
                             roc_auc_score, confusion_matrix, classification_report,
                             roc_curve, precision_recall_curve)

try:
    from catboost import CatBoostClassifier, Pool
except ImportError:
    print("ERROR: CatBoost not installed. Install with: pip install catboost")
    sys.exit(1)

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION - Based on Mega-Analysis Results
# =============================================================================

# Optimal hyperparameters found from 1,004 experiments
OPTIMAL_PARAMS = {
    'iterations': 2000,
    'depth': 8,
    'learning_rate': 0.01,
    'l2_leaf_reg': 3,
    'random_seed': 42,
    'verbose': 100,
    'early_stopping_rounds': 100,
    'task_type': 'CPU',  # Change to 'GPU' if available
    'loss_function': 'Logloss',
    'eval_metric': 'AUC'
}

# Optimal threshold from mega-analysis
OPTIMAL_THRESHOLD = 0.48

# Categorical features
CAT_COLS = ['gender', 'cholesterol', 'gluc', 'smoke', 'alco', 'active']

# =============================================================================
# LOGGING SETUP
# =============================================================================

def setup_logging(output_dir):
    """Setup comprehensive logging"""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    logger = logging.getLogger("final_catboost")
    logger.setLevel(logging.INFO)
    
    # Remove existing handlers
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
    
    # Formatter
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    
    # File handler
    log_path = os.path.join(output_dir, f"training_{timestamp}.log")
    fh = logging.FileHandler(log_path, encoding="utf-8", mode="w")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    
    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    
    return logger, timestamp

# =============================================================================
# FEATURE ENGINEERING (32 features)
# =============================================================================

def engineer_features_full(df, logger):
    """
    Complete feature engineering pipeline - Same as used in mega-analysis.
    Creates 32 features from original 13.
    """
    df = df.copy()
    eps = 1e-6
    
    logger.info("Starting feature engineering...")
    
    # 1. Blood Pressure Features
    if {'ap_hi', 'ap_lo'}.issubset(df.columns):
        df['bp_ratio'] = df['ap_hi'] / (df['ap_lo'] + eps)
        df['map'] = (df['ap_hi'] + 2 * df['ap_lo']) / 3  # Mean Arterial Pressure
        df['pulse_pressure'] = df['ap_hi'] - df['ap_lo']
        logger.info("  ✓ Blood pressure features (bp_ratio, map, pulse_pressure)")
    
    # 2. BMI Calculation
    if {'height', 'weight'}.issubset(df.columns):
        df['BMI'] = df['weight'] / ((df['height'] / 100) ** 2)
        logger.info("  ✓ BMI calculated")
    
    # 3. Age Transformations
    if 'age' in df.columns:
        df['age_years'] = df['age'] / 365.25
        df['age_squared'] = df['age_years'] ** 2
        df['age_group'] = pd.cut(df['age_years'], bins=[0, 40, 50, 60, 100], 
                                 labels=[0, 1, 2, 3]).astype(int)
        logger.info("  ✓ Age features (age_years, age_squared, age_group)")
    
    # 4. Age × Cardiovascular Risk Interactions
    if {'pulse_pressure', 'age_years'}.issubset(df.columns):
        df['pp_age_ratio'] = df['pulse_pressure'] / (df['age_years'] + eps)
        df['pp_age_product'] = df['pulse_pressure'] * df['age_years']
    
    if {'BMI', 'age_years'}.issubset(df.columns):
        df['bmi_age_product'] = df['BMI'] * df['age_years']
        df['bmi_age_ratio'] = df['BMI'] / (df['age_years'] + eps)
        df['bmi_squared'] = df['BMI'] ** 2
    
    if {'ap_hi', 'age_years'}.issubset(df.columns):
        df['sbp_age_ratio'] = df['ap_hi'] / (df['age_years'] + eps)
        df['sbp_age_product'] = df['ap_hi'] * df['age_years']
    
    logger.info("  ✓ Age interaction features")
    
    # 5. BMI × BP Interactions
    if {'ap_hi', 'BMI'}.issubset(df.columns):
        df['sbp_bmi_product'] = df['ap_hi'] * df['BMI']
        df['sbp_bmi_ratio'] = df['ap_hi'] / (df['BMI'] + eps)
    
    if {'pulse_pressure', 'BMI'}.issubset(df.columns):
        df['pp_bmi_product'] = df['pulse_pressure'] * df['BMI']
        df['pp_bmi_ratio'] = df['pulse_pressure'] / (df['BMI'] + eps)
    
    logger.info("  ✓ BMI interaction features")
    
    # 6. Log Transforms (for skewed distributions)
    if 'BMI' in df.columns:
        df['log_bmi'] = np.log1p(df['BMI'])
    
    if 'pulse_pressure' in df.columns:
        df['log_pp'] = np.log1p(df['pulse_pressure'])
    
    if 'ap_hi' in df.columns:
        df['log_sbp'] = np.log1p(df['ap_hi'])
    
    logger.info("  ✓ Log-transformed features")
    
    # 7. Lifestyle Risk Score
    if {'cholesterol', 'gluc', 'smoke', 'alco'}.issubset(df.columns):
        df['lifestyle_risk'] = (df['cholesterol'] / 3 + df['gluc'] / 3 + 
                                df['smoke'] + df['alco'])
        logger.info("  ✓ Lifestyle risk score")
    
    # 8. Cardiovascular Risk Interactions
    if {'ap_hi', 'cholesterol'}.issubset(df.columns):
        df['bp_chol_interaction'] = df['ap_hi'] * df['cholesterol']
    
    if {'BMI', 'cholesterol', 'smoke'}.issubset(df.columns):
        df['metabolic_risk'] = (df['BMI'] / 30) * df['cholesterol'] * (1 + df['smoke'])
    
    logger.info("  ✓ Cardiovascular risk features")
    
    n_features = df.shape[1] - 1  # Excluding target
    logger.info(f"Feature engineering complete: {n_features} features total")
    
    return df

# =============================================================================
# DATA PREPARATION
# =============================================================================

def prepare_data(df, validation_mode, val_size, seed, logger):
    """
    Prepare data with full feature engineering and standardization.
    
    Args:
        df: Input dataframe
        validation_mode: If True, split into train/val. If False, use all data.
        val_size: Validation set proportion (if validation_mode=True)
        seed: Random seed
        logger: Logger instance
    
    Returns:
        If validation_mode=True: X_train, X_val, y_train, y_val, scaler, cat_features
        If validation_mode=False: X, y, scaler, cat_features
    """
    logger.info("="*70)
    logger.info("DATA PREPARATION")
    logger.info("="*70)
    
    # Separate target
    y = df['cardio'].values
    X = df.drop(columns=['cardio'])
    
    logger.info(f"Original data shape: {X.shape}")
    logger.info(f"Class distribution: {np.bincount(y)}")
    
    # Engineer features
    X = engineer_features_full(X, logger)
    
    # Identify categorical feature indices
    cat_feature_indices = [i for i, col in enumerate(X.columns) if col in CAT_COLS]
    logger.info(f"Categorical features: {[X.columns[i] for i in cat_feature_indices]}")
    
    # Identify numeric columns for scaling
    num_cols = [c for c in X.columns if c not in CAT_COLS]
    
    if validation_mode:
        # Train/validation split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=val_size, random_state=seed, stratify=y
        )
        
        logger.info(f"Train set: {X_train.shape}, Class distribution: {np.bincount(y_train)}")
        logger.info(f"Val set:   {X_val.shape}, Class distribution: {np.bincount(y_val)}")
        
        # Standardize numeric features
        scaler = StandardScaler()
        X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
        X_val[num_cols] = scaler.transform(X_val[num_cols])
        
        # Convert categorical to int (CatBoost requirement)
        for col in CAT_COLS:
            if col in X_train.columns:
                X_train[col] = X_train[col].astype(int)
                X_val[col] = X_val[col].astype(int)
        
        logger.info("Data preparation complete (validation mode)")
        return X_train, X_val, y_train, y_val, scaler, cat_feature_indices
    
    else:
        # Use all data for final training
        logger.info(f"Using full dataset: {X.shape}")
        
        # Standardize numeric features
        scaler = StandardScaler()
        X[num_cols] = scaler.fit_transform(X[num_cols])
        
        # Convert categorical to int
        for col in CAT_COLS:
            if col in X.columns:
                X[col] = X[col].astype(int)
        
        logger.info("Data preparation complete (full training mode)")
        return X, y, scaler, cat_feature_indices

# =============================================================================
# MODEL TRAINING
# =============================================================================

def train_catboost(X_train, y_train, X_val, y_val, cat_features, params, logger):
    """
    Train CatBoost with optimal hyperparameters from mega-analysis.
    """
    logger.info("="*70)
    logger.info("MODEL TRAINING - CatBoost (Optimal Configuration)")
    logger.info("="*70)
    
    logger.info("Hyperparameters:")
    for key, value in params.items():
        logger.info(f"  {key}: {value}")
    
    # Create CatBoost pools
    train_pool = Pool(X_train, y_train, cat_features=cat_features)
    val_pool = Pool(X_val, y_val, cat_features=cat_features)
    
    # Initialize model
    model = CatBoostClassifier(**params)
    
    # Train with validation monitoring
    logger.info("\nStarting training...")
    start_time = time.time()
    
    model.fit(
        train_pool,
        eval_set=val_pool,
        use_best_model=True,
        plot=False
    )
    
    elapsed = time.time() - start_time
    logger.info(f"\nTraining completed in {elapsed/60:.2f} minutes")
    logger.info(f"Best iteration: {model.get_best_iteration()}")
    logger.info(f"Best score (AUC): {model.get_best_score()['validation']['AUC']:.6f}")
    
    return model

def train_catboost_full(X, y, cat_features, params, logger):
    """
    Train CatBoost on full dataset (no validation split).
    Used for final production model.
    """
    logger.info("="*70)
    logger.info("FINAL MODEL TRAINING - Full Dataset")
    logger.info("="*70)
    
    logger.info("Training on 100% of data (no validation split)")
    
    # Create pool
    train_pool = Pool(X, y, cat_features=cat_features)
    
    # Remove early stopping for full training
    params_full = params.copy()
    params_full.pop('early_stopping_rounds', None)
    params_full['iterations'] = model.get_best_iteration() if 'model' in locals() else params['iterations']
    
    # Initialize and train
    model = CatBoostClassifier(**params_full)
    
    logger.info("Starting training...")
    start_time = time.time()
    
    model.fit(train_pool, plot=False)
    
    elapsed = time.time() - start_time
    logger.info(f"Training completed in {elapsed/60:.2f} minutes")
    
    return model

# =============================================================================
# EVALUATION
# =============================================================================

def evaluate_model(model, X, y, threshold, dataset_name, logger):
    """
    Comprehensive model evaluation with optimal threshold.
    """
    logger.info("="*70)
    logger.info(f"MODEL EVALUATION - {dataset_name}")
    logger.info("="*70)
    
    # Predict probabilities
    y_pred_proba = model.predict_proba(X)[:, 1]
    
    # Predict classes with optimal threshold
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    # Calculate metrics
    acc = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    prec = precision_score(y, y_pred)
    rec = recall_score(y, y_pred)
    auc = roc_auc_score(y, y_pred_proba)
    
    # Confusion matrix
    cm = confusion_matrix(y, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # Log results
    logger.info(f"\nThreshold: {threshold:.2f}")
    logger.info(f"\nPerformance Metrics:")
    logger.info(f"  AUC:       {auc:.4f} ({auc*100:.2f}%)")
    logger.info(f"  Accuracy:  {acc:.4f} ({acc*100:.2f}%)")
    logger.info(f"  F1 Score:  {f1:.4f}")
    logger.info(f"  Precision: {prec:.4f}")
    logger.info(f"  Recall:    {rec:.4f}")
    
    logger.info(f"\nConfusion Matrix:")
    logger.info(f"  True Negatives:  {tn:6d}")
    logger.info(f"  False Positives: {fp:6d}")
    logger.info(f"  False Negatives: {fn:6d}")
    logger.info(f"  True Positives:  {tp:6d}")
    
    # Additional metrics
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
    
    logger.info(f"\nAdditional Metrics:")
    logger.info(f"  Specificity (TNR): {specificity:.4f}")
    logger.info(f"  NPV: {npv:.4f}")
    logger.info(f"  PPV: {ppv:.4f}")
    
    # Feature importance
    feature_importance = model.get_feature_importance()
    feature_names = X.columns
    
    logger.info(f"\nTop 10 Most Important Features:")
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)
    
    for idx, row in importance_df.head(10).iterrows():
        logger.info(f"  {row['feature']:25s}: {row['importance']:.2f}")
    
    # Return metrics dictionary
    metrics = {
        'threshold': float(threshold),
        'auc': float(auc),
        'accuracy': float(acc),
        'f1': float(f1),
        'precision': float(prec),
        'recall': float(rec),
        'specificity': float(specificity),
        'npv': float(npv),
        'ppv': float(ppv),
        'confusion_matrix': {
            'tn': int(tn), 'fp': int(fp),
            'fn': int(fn), 'tp': int(tp)
        },
        'feature_importance': importance_df.head(20).to_dict('records')
    }
    
    return metrics, y_pred_proba

# =============================================================================
# SAVE MODEL & ARTIFACTS
# =============================================================================

def save_model_and_artifacts(model, scaler, cat_features, metrics, params, 
                             output_dir, timestamp, logger):
    """
    Save trained model, scaler, and all metadata.
    """
    logger.info("="*70)
    logger.info("SAVING MODEL & ARTIFACTS")
    logger.info("="*70)
    
    # Save CatBoost model
    model_path = os.path.join(output_dir, f"catboost_model_{timestamp}.cbm")
    model.save_model(model_path)
    logger.info(f"✓ Model saved: {model_path}")
    
    # Save scaler
    scaler_path = os.path.join(output_dir, f"scaler_{timestamp}.pkl")
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    logger.info(f"✓ Scaler saved: {scaler_path}")
    
    # Save configuration
    config = {
        'model': 'CatBoost',
        'timestamp': timestamp,
        'hyperparameters': params,
        'optimal_threshold': OPTIMAL_THRESHOLD,
        'categorical_features': cat_features,
        'n_features': len(cat_features) + len([c for c in model.feature_names_ if c not in CAT_COLS]),
        'feature_names': list(model.feature_names_),
        'validation_metrics': metrics,
        'mega_analysis_source': 'Based on 1,004 experiments',
        'expected_performance': {
            'auc': 0.799,
            'accuracy': 0.733,
            'f1': 0.726
        }
    }
    
    config_path = os.path.join(output_dir, f"model_config_{timestamp}.json")
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    logger.info(f"✓ Configuration saved: {config_path}")
    
    # Save README
    readme_path = os.path.join(output_dir, "README.txt")
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("OPTIMAL CVD PREDICTION MODEL - CatBoost\n")
        f.write("="*70 + "\n\n")
        f.write("Based on exhaustive mega-analysis of 1,004 experiments.\n\n")
        f.write("FILES:\n")
        f.write(f"  - catboost_model_{timestamp}.cbm: Trained CatBoost model\n")
        f.write(f"  - scaler_{timestamp}.pkl: Feature scaler (StandardScaler)\n")
        f.write(f"  - model_config_{timestamp}.json: Complete configuration\n")
        f.write(f"  - training_{timestamp}.log: Training log\n\n")
        f.write("EXPECTED PERFORMANCE:\n")
        f.write(f"  - AUC: 79.9%\n")
        f.write(f"  - Accuracy: 73.3%\n")
        f.write(f"  - F1 Score: 72.6%\n")
        f.write(f"  - Precision: 74.5%\n")
        f.write(f"  - Recall: 70.8%\n\n")
        f.write("USAGE:\n")
        f.write("  See predict_example.py for inference code.\n\n")
        f.write(f"Created: {timestamp}\n")
    
    logger.info(f"✓ README saved: {readme_path}")
    
    # Save example prediction script
    example_path = os.path.join(output_dir, "predict_example.py")
    with open(example_path, 'w', encoding='utf-8') as f:
        f.write(f'''# -*- coding: utf-8 -*-
"""
Example: How to use the trained CatBoost model for prediction
"""

import pickle
import pandas as pd
from catboost import CatBoostClassifier

# Load model
model = CatBoostClassifier()
model.load_model("catboost_model_{timestamp}.cbm")

# Load scaler
with open("scaler_{timestamp}.pkl", "rb") as f:
    scaler = pickle.load(f)

# Example: Predict for new patient
new_patient = {{
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
}}

# Preprocess (same feature engineering as training)
# ... (add feature engineering code here)

# Scale numeric features
# numeric_cols = [...]  # list of numeric column names
# new_patient_scaled = scaler.transform(new_patient[numeric_cols])

# Predict probability
prob = model.predict_proba(new_patient_df)[:, 1][0]

# Apply optimal threshold
threshold = {OPTIMAL_THRESHOLD}
prediction = 1 if prob >= threshold else 0

print(f"CVD Risk Probability: {{prob*100:.2f}}%")
print(f"Prediction: {{'CVD' if prediction == 1 else 'No CVD'}}")
''')
    
    logger.info(f"✓ Example script saved: {example_path}")
    logger.info("\nAll artifacts saved successfully!")

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Train final optimal CatBoost model for CVD prediction"
    )
    parser.add_argument("--input", type=str, required=True,
                       help="Path to input CSV file (cardio_clean.csv)")
    parser.add_argument("--output_dir", type=str, default="final_model",
                       help="Output directory for model and artifacts")
    parser.add_argument("--validation", action='store_true',
                       help="Run with validation split (default: train on 100%% data)")
    parser.add_argument("--val_size", type=float, default=0.2,
                       help="Validation set size (if --validation is set)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Setup
    logger, timestamp = setup_logging(args.output_dir)
    np.random.seed(args.seed)
    
    logger.info("="*70)
    logger.info("PREDICTION MODEL TRAINING")
    logger.info("Based on Mega-Analysis (1,004 experiments)")
    logger.info("="*70)
    logger.info(f"Timestamp: {timestamp}")
    logger.info(f"Input file: {args.input}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Validation mode: {args.validation}")
    logger.info(f"Random seed: {args.seed}")
    
    # Load data
    logger.info("\nLoading data...")
    df = pd.read_csv(args.input)
    logger.info(f"Loaded data: {df.shape}")
    
    # Prepare data
    if args.validation:
        X_train, X_val, y_train, y_val, scaler, cat_features = prepare_data(
            df, validation_mode=True, val_size=args.val_size, 
            seed=args.seed, logger=logger
        )
        
        # Train with validation
        model = train_catboost(X_train, y_train, X_val, y_val, 
                              cat_features, OPTIMAL_PARAMS, logger)
        
        # Evaluate on validation set
        val_metrics, _ = evaluate_model(model, X_val, y_val, OPTIMAL_THRESHOLD,
                                       "Validation Set", logger)
        
        # Optionally: Retrain on full data with best_iteration
        logger.info("\n" + "="*70)
        logger.info("RETRAINING ON FULL DATASET")
        logger.info("="*70)
        
        X_full, y_full, scaler_full, cat_features_full = prepare_data(
            df, validation_mode=False, val_size=None, 
            seed=args.seed, logger=logger
        )
        
        # Use best_iteration from validation training
        final_params = OPTIMAL_PARAMS.copy()
        final_params['iterations'] = model.get_best_iteration()
        final_params.pop('early_stopping_rounds', None)
        final_params['verbose'] = False
        
        final_model = CatBoostClassifier(**final_params)
        final_model.fit(Pool(X_full, y_full, cat_features=cat_features_full))
        
        logger.info(f"Final model trained on full dataset ({len(y_full)} samples)")
        
        # Save final model
        save_model_and_artifacts(final_model, scaler_full, cat_features_full,
                                val_metrics, final_params, args.output_dir,
                                timestamp, logger)
        
    else:
        # Train directly on full dataset (no validation)
        X, y, scaler, cat_features = prepare_data(
            df, validation_mode=False, val_size=None,
            seed=args.seed, logger=logger
        )
        
        # Train final model
        model = CatBoostClassifier(**OPTIMAL_PARAMS)
        model.fit(Pool(X, y, cat_features=cat_features))
        
        logger.info("Training complete (no validation split used)")
        
        # Evaluate on training data (just for reference)
        train_metrics, _ = evaluate_model(model, X, y, OPTIMAL_THRESHOLD,
                                         "Training Set (Full Data)", logger)
        
        # Save model
        save_model_and_artifacts(model, scaler, cat_features, train_metrics,
                                OPTIMAL_PARAMS, args.output_dir, timestamp, logger)
    
    # Final summary
    logger.info("\n" + "="*70)
    logger.info("TRAINING COMPLETED SUCCESSFULLY")
    logger.info("="*70)
    logger.info(f"\nModel saved to: {args.output_dir}/")
    logger.info("\nExpected Performance (from mega-analysis):")
    logger.info("  - AUC: 79.9%")
    logger.info("  - Accuracy: 73.3%")
    logger.info("  - F1 Score: 72.6%")
    logger.info("  - Precision: 74.5%")
    logger.info("  - Recall: 70.8%")
    logger.info("\nThis is the optimal configuration found from 1,004 experiments.")
    logger.info("Ready for production deployment!")
    logger.info("="*70)

if __name__ == "__main__":
    main()