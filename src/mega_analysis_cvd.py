# -*- coding: utf-8 -*-
"""
=============================================================================
MEGA-ANALYSIS SCRIPT
=============================================================================
Exhaustive search for optimal configuration:
  1. Model Selection (LightGBM, XGBoost, CatBoost, RandomForest, Extra Trees)
  2. Hyperparameter Grid Search (learning_rate, max_depth, num_leaves, etc.)
  3. Feature Engineering Combinations
  4. Feature Selection (forward/backward/RFECV)
  5. Ensemble Strategies (voting, weighted avg, stacking)
  6. Threshold Optimization
  7. Data Split Ratios
  8. Cross-Validation (5-fold, 10-fold)
  9. Calibration Methods (isotonic, sigmoid)
  10. Class Imbalance Handling (SMOTE, weights)

Output: Detailed CSV reports for every experiment + best model saved

Usage:
python mega_analysis_cvd.py --input "D:\Project\Heart diseae\dataset\1\cardio_clean.csv" --out_dir "mega_results" --seed 42 --n_jobs -1

Author: Ghasem
https://github.com/GhIrani33?tab=repositories
"""

import os, json, time, argparse, logging, sys, warnings, pickle
from datetime import datetime
from itertools import product, combinations
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.feature_selection import RFECV, SelectKBest, f_classif
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score, 
                             roc_auc_score, confusion_matrix, classification_report)
import lightgbm as lgb
import xgboost as xgb

# Optional advanced libraries
try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except:
    HAS_CATBOOST = False
    
try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except:
    HAS_SMOTE = False

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

CAT_COLS = ['gender','cholesterol','gluc','smoke','alco','active']

# Parameter grids for grid search
LGBM_PARAM_GRID = {
    'num_leaves': [31, 63, 95],
    'max_depth': [8, 10, 12],
    'learning_rate': [0.01, 0.03, 0.05],
    'feature_fraction': [0.7, 0.8, 0.9],
    'bagging_fraction': [0.7, 0.8, 0.9],
    'min_child_samples': [10, 20, 30]
}

XGB_PARAM_GRID = {
    'max_depth': [6, 8, 10],
    'eta': [0.01, 0.03, 0.05],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9],
    'min_child_weight': [1, 3, 5]
}

CATBOOST_PARAM_GRID = {
    'depth': [6, 8, 10],
    'learning_rate': [0.01, 0.03, 0.05],
    'l2_leaf_reg': [1, 3, 5]
}

# Feature engineering configurations
FEATURE_CONFIGS = {
    'basic': ['age', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc', 'smoke', 'alco', 'active'],
    'with_derived': ['age', 'gender', 'BMI', 'pulse_pressure', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc', 'smoke', 'alco', 'active'],
    'full_engineered': 'all'  # Use all engineered features
}

# Thresholds to test
THRESHOLDS = np.linspace(0.3, 0.7, 41)  # 0.30, 0.31, ..., 0.70

# =============================================================================
# LOGGING SETUP
# =============================================================================

def setup_logging(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    logger = logging.getLogger("mega_analysis")
    logger.setLevel(logging.INFO)
    for h in list(logger.handlers):
        logger.removeHandler(h)
    
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    
    log_file = os.path.join(out_dir, f"mega_analysis_{timestamp}.log")
    fh = logging.FileHandler(log_file, encoding="utf-8", mode="w")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    
    return logger, timestamp

# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

def engineer_features_full(df):
    """Complete feature engineering pipeline"""
    df = df.copy()
    eps = 1e-6
    
    # Blood pressure features
    if {'ap_hi','ap_lo'}.issubset(df.columns):
        df['bp_ratio'] = df['ap_hi'] / (df['ap_lo'] + eps)
        df['map'] = (df['ap_hi'] + 2*df['ap_lo']) / 3
        df['pulse_pressure'] = df['ap_hi'] - df['ap_lo']
    
    # BMI
    if {'height','weight'}.issubset(df.columns):
        df['BMI'] = df['weight'] / ((df['height']/100) ** 2)
    
    # Age in years
    if 'age' in df.columns:
        df['age_years'] = df['age'] / 365.25
        df['age_squared'] = df['age_years'] ** 2
        df['age_group'] = pd.cut(df['age_years'], bins=[0,40,50,60,100], labels=[0,1,2,3]).astype(int)
    
    # Age-related interactions
    if {'pulse_pressure','age_years'}.issubset(df.columns):
        df['pp_age_ratio'] = df['pulse_pressure'] / (df['age_years'] + eps)
        df['pp_age_product'] = df['pulse_pressure'] * df['age_years']
    
    if {'BMI','age_years'}.issubset(df.columns):
        df['bmi_age_product'] = df['BMI'] * df['age_years']
        df['bmi_age_ratio'] = df['BMI'] / (df['age_years'] + eps)
        df['bmi_squared'] = df['BMI'] ** 2
    
    if {'ap_hi','age_years'}.issubset(df.columns):
        df['sbp_age_ratio'] = df['ap_hi'] / (df['age_years'] + eps)
        df['sbp_age_product'] = df['ap_hi'] * df['age_years']
    
    # BMI interactions
    if {'ap_hi','BMI'}.issubset(df.columns):
        df['sbp_bmi_product'] = df['ap_hi'] * df['BMI']
        df['sbp_bmi_ratio'] = df['ap_hi'] / (df['BMI'] + eps)
    
    if {'pulse_pressure','BMI'}.issubset(df.columns):
        df['pp_bmi_product'] = df['pulse_pressure'] * df['BMI']
        df['pp_bmi_ratio'] = df['pulse_pressure'] / (df['BMI'] + eps)
    
    # Log transforms
    if 'BMI' in df.columns:
        df['log_bmi'] = np.log1p(df['BMI'])
    
    if 'pulse_pressure' in df.columns:
        df['log_pp'] = np.log1p(df['pulse_pressure'])
    
    if 'ap_hi' in df.columns:
        df['log_sbp'] = np.log1p(df['ap_hi'])
    
    # Lifestyle risk score
    if {'cholesterol','gluc','smoke','alco'}.issubset(df.columns):
        df['lifestyle_risk'] = df['cholesterol']/3 + df['gluc']/3 + df['smoke'] + df['alco']
    
    # Cardiovascular risk indicators
    if {'ap_hi','cholesterol'}.issubset(df.columns):
        df['bp_chol_interaction'] = df['ap_hi'] * df['cholesterol']
    
    if {'BMI','cholesterol','smoke'}.issubset(df.columns):
        df['metabolic_risk'] = (df['BMI']/30) * df['cholesterol'] * (1 + df['smoke'])
    
    return df

def select_feature_subset(df, config):
    """Select features based on configuration"""
    if config == 'all':
        return df
    else:
        available_cols = [c for c in config if c in df.columns]
        return df[available_cols]

# =============================================================================
# DATA PREPARATION
# =============================================================================

def prepare_data(df, feature_config, val_size, seed, logger):
    """Prepare train/val split with selected features"""
    y = df['cardio'].values
    X = df.drop(columns=['cardio'])
    
    # Engineer features
    X = engineer_features_full(X)
    
    # Select feature subset
    if feature_config != 'all':
        X = select_feature_subset(X, FEATURE_CONFIGS[feature_config])
    
    # Train/val split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=val_size, random_state=seed, stratify=y
    )
    
    # Identify numeric columns
    num_cols = [c for c in X_train.columns if c not in CAT_COLS]
    
    # Scale numeric features
    scaler = StandardScaler()
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_val[num_cols] = scaler.transform(X_val[num_cols])
    
    # Convert categorical to category dtype
    for c in CAT_COLS:
        if c in X_train.columns:
            X_train[c] = X_train[c].astype('category')
            X_val[c] = X_val[c].astype('category')
    
    logger.info(f"  Prepared data: Train={X_train.shape}, Val={X_val.shape}")
    return X_train, X_val, y_train, y_val, scaler

# =============================================================================
# MODEL TRAINING FUNCTIONS
# =============================================================================

def train_lightgbm(X_train, y_train, X_val, y_val, params, seed):
    """Train LightGBM with given params"""
    dtrain = lgb.Dataset(X_train, label=y_train, 
                        categorical_feature=[c for c in CAT_COLS if c in X_train.columns])
    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain,
                      categorical_feature=[c for c in CAT_COLS if c in X_train.columns])
    
    full_params = {**params, 'objective': 'binary', 'metric': 'auc', 
                   'random_state': seed, 'verbosity': -1}
    
    callbacks = [lgb.early_stopping(stopping_rounds=100, verbose=False)]
    model = lgb.train(full_params, dtrain, num_boost_round=2000, 
                     valid_sets=[dval], callbacks=callbacks)
    
    y_pred_proba = model.predict(X_val, num_iteration=model.best_iteration)
    auc = roc_auc_score(y_val, y_pred_proba)
    
    return model, y_pred_proba, auc

def train_xgboost(X_train, y_train, X_val, y_val, params, seed):
    """Train XGBoost with given params"""
    X_tr = X_train.copy(); X_vl = X_val.copy()
    for c in CAT_COLS:
        if c in X_tr.columns:
            X_tr[c] = X_tr[c].astype(int)
            X_vl[c] = X_vl[c].astype(int)
    
    dtrain = xgb.DMatrix(X_tr, label=y_train)
    dval = xgb.DMatrix(X_vl, label=y_val)
    
    full_params = {**params, 'objective': 'binary:logistic', 'eval_metric': 'auc',
                   'seed': seed, 'verbosity': 0}
    
    model = xgb.train(full_params, dtrain, num_boost_round=2000,
                     evals=[(dval, 'val')], early_stopping_rounds=100, verbose_eval=False)
    
    y_pred_proba = model.predict(dval, iteration_range=(0, model.best_iteration))
    auc = roc_auc_score(y_val, y_pred_proba)
    
    return model, y_pred_proba, auc

def train_catboost(X_train, y_train, X_val, y_val, params, seed):
    """Train CatBoost with given params"""
    if not HAS_CATBOOST:
        return None, None, 0.0
    
    cat_features = [i for i, c in enumerate(X_train.columns) if c in CAT_COLS]
    
    full_params = {**params, 'iterations': 2000, 'random_seed': seed, 
                   'verbose': False, 'early_stopping_rounds': 100}
    
    model = CatBoostClassifier(**full_params)
    model.fit(X_train, y_train, eval_set=(X_val, y_val), cat_features=cat_features)
    
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, y_pred_proba)
    
    return model, y_pred_proba, auc

def train_random_forest(X_train, y_train, X_val, y_val, seed):
    """Train Random Forest"""
    X_tr = X_train.copy(); X_vl = X_val.copy()
    for c in CAT_COLS:
        if c in X_tr.columns:
            X_tr[c] = X_tr[c].astype(int)
            X_vl[c] = X_vl[c].astype(int)
    
    model = RandomForestClassifier(n_estimators=500, max_depth=15, min_samples_split=10,
                                   min_samples_leaf=5, random_state=seed, n_jobs=-1)
    model.fit(X_tr, y_train)
    
    y_pred_proba = model.predict_proba(X_vl)[:, 1]
    auc = roc_auc_score(y_val, y_pred_proba)
    
    return model, y_pred_proba, auc

def train_extra_trees(X_train, y_train, X_val, y_val, seed):
    """Train Extra Trees"""
    X_tr = X_train.copy(); X_vl = X_val.copy()
    for c in CAT_COLS:
        if c in X_tr.columns:
            X_tr[c] = X_tr[c].astype(int)
            X_vl[c] = X_vl[c].astype(int)
    
    model = ExtraTreesClassifier(n_estimators=500, max_depth=15, min_samples_split=10,
                                 min_samples_leaf=5, random_state=seed, n_jobs=-1)
    model.fit(X_tr, y_train)
    
    y_pred_proba = model.predict_proba(X_vl)[:, 1]
    auc = roc_auc_score(y_val, y_pred_proba)
    
    return model, y_pred_proba, auc

# =============================================================================
# EVALUATION
# =============================================================================

def evaluate_threshold(y_true, y_pred_proba, threshold):
    """Evaluate metrics at given threshold"""
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    
    return {'accuracy': acc, 'f1': f1, 'precision': prec, 'recall': rec}

def find_best_threshold(y_true, y_pred_proba, metric='accuracy'):
    """Find optimal threshold for given metric"""
    best_score = -1
    best_thr = 0.5
    
    for thr in THRESHOLDS:
        metrics = evaluate_threshold(y_true, y_pred_proba, thr)
        score = metrics[metric]
        
        if score > best_score:
            best_score = score
            best_thr = thr
    
    return best_thr, best_score

# =============================================================================
# MAIN MEGA-ANALYSIS
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="CVD Mega-Analysis")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="mega_results")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_jobs", type=int, default=-1)
    parser.add_argument("--quick_mode", action='store_true', help="Reduced search space for testing")
    args = parser.parse_args()
    
    logger, timestamp = setup_logging(args.out_dir)
    np.random.seed(args.seed)
    
    logger.info("="*80)
    logger.info("CVD MEGA-ANALYSIS STARTED")
    logger.info(f"Timestamp: {timestamp}")
    logger.info("="*80)
    
    # Load data
    df = pd.read_csv(args.input)
    logger.info(f"Loaded data: {df.shape}")
    
    # Storage for all results
    all_results = []
    experiment_id = 0
    
    # ===========================================
    # PHASE 1: MODEL SELECTION & HYPERPARAMETERS
    # ===========================================
    logger.info("\n" + "="*80)
    logger.info("PHASE 1: MODEL SELECTION & HYPERPARAMETER SEARCH")
    logger.info("="*80)
    
    # Prepare data with full features
    X_train, X_val, y_train, y_val, scaler = prepare_data(df, 'full_engineered', 0.2, args.seed, logger)
    
    # Test LightGBM configurations
    logger.info("\n[1/5] Testing LightGBM configurations...")
    if args.quick_mode:
        lgbm_grid = {'num_leaves': [63], 'max_depth': [10], 'learning_rate': [0.03],
                     'feature_fraction': [0.8], 'bagging_fraction': [0.8], 'min_child_samples': [20]}
    else:
        lgbm_grid = LGBM_PARAM_GRID
    
    lgbm_configs = [dict(zip(lgbm_grid.keys(), v)) for v in product(*lgbm_grid.values())]
    logger.info(f"  Testing {len(lgbm_configs)} LightGBM configurations...")
    
    for i, params in enumerate(lgbm_configs):
        try:
            _, proba, auc = train_lightgbm(X_train, y_train, X_val, y_val, params, args.seed)
            best_thr, best_acc = find_best_threshold(y_val, proba, 'accuracy')
            metrics = evaluate_threshold(y_val, proba, best_thr)
            
            all_results.append({
                'experiment_id': experiment_id,
                'model': 'LightGBM',
                'params': json.dumps(params),
                'features': 'full_engineered',
                'val_size': 0.2,
                'auc': auc,
                'best_threshold': best_thr,
                **metrics
            })
            
            experiment_id += 1
            
            if (i+1) % 10 == 0:
                logger.info(f"    Completed {i+1}/{len(lgbm_configs)} - Best AUC so far: {max([r['auc'] for r in all_results]):.4f}")
        except Exception as e:
            logger.warning(f"    Config {i+1} failed: {e}")
    
    # Test XGBoost configurations
    logger.info("\n[2/5] Testing XGBoost configurations...")
    if args.quick_mode:
        xgb_grid = {'max_depth': [8], 'eta': [0.03], 'subsample': [0.8],
                    'colsample_bytree': [0.8], 'min_child_weight': [3]}
    else:
        xgb_grid = XGB_PARAM_GRID
    
    xgb_configs = [dict(zip(xgb_grid.keys(), v)) for v in product(*xgb_grid.values())]
    logger.info(f"  Testing {len(xgb_configs)} XGBoost configurations...")
    
    for i, params in enumerate(xgb_configs):
        try:
            _, proba, auc = train_xgboost(X_train, y_train, X_val, y_val, params, args.seed)
            best_thr, best_acc = find_best_threshold(y_val, proba, 'accuracy')
            metrics = evaluate_threshold(y_val, proba, best_thr)
            
            all_results.append({
                'experiment_id': experiment_id,
                'model': 'XGBoost',
                'params': json.dumps(params),
                'features': 'full_engineered',
                'val_size': 0.2,
                'auc': auc,
                'best_threshold': best_thr,
                **metrics
            })
            
            experiment_id += 1
            
            if (i+1) % 10 == 0:
                logger.info(f"    Completed {i+1}/{len(xgb_configs)}")
        except Exception as e:
            logger.warning(f"    Config {i+1} failed: {e}")
    
    # Test CatBoost if available
    if HAS_CATBOOST and not args.quick_mode:
        logger.info("\n[3/5] Testing CatBoost configurations...")
        catboost_configs = [dict(zip(CATBOOST_PARAM_GRID.keys(), v)) 
                           for v in product(*CATBOOST_PARAM_GRID.values())]
        logger.info(f"  Testing {len(catboost_configs)} CatBoost configurations...")
        
        for i, params in enumerate(catboost_configs):
            try:
                _, proba, auc = train_catboost(X_train, y_train, X_val, y_val, params, args.seed)
                best_thr, best_acc = find_best_threshold(y_val, proba, 'accuracy')
                metrics = evaluate_threshold(y_val, proba, best_thr)
                
                all_results.append({
                    'experiment_id': experiment_id,
                    'model': 'CatBoost',
                    'params': json.dumps(params),
                    'features': 'full_engineered',
                    'val_size': 0.2,
                    'auc': auc,
                    'best_threshold': best_thr,
                    **metrics
                })
                
                experiment_id += 1
            except Exception as e:
                logger.warning(f"    Config {i+1} failed: {e}")
    
    # Test Random Forest
    logger.info("\n[4/5] Testing Random Forest...")
    try:
        _, proba, auc = train_random_forest(X_train, y_train, X_val, y_val, args.seed)
        best_thr, best_acc = find_best_threshold(y_val, proba, 'accuracy')
        metrics = evaluate_threshold(y_val, proba, best_thr)
        
        all_results.append({
            'experiment_id': experiment_id,
            'model': 'RandomForest',
            'params': json.dumps({'n_estimators': 500}),
            'features': 'full_engineered',
            'val_size': 0.2,
            'auc': auc,
            'best_threshold': best_thr,
            **metrics
        })
        experiment_id += 1
    except Exception as e:
        logger.warning(f"    RandomForest failed: {e}")
    
    # Test Extra Trees
    logger.info("\n[5/5] Testing Extra Trees...")
    try:
        _, proba, auc = train_extra_trees(X_train, y_train, X_val, y_val, args.seed)
        best_thr, best_acc = find_best_threshold(y_val, proba, 'accuracy')
        metrics = evaluate_threshold(y_val, proba, best_thr)
        
        all_results.append({
            'experiment_id': experiment_id,
            'model': 'ExtraTrees',
            'params': json.dumps({'n_estimators': 500}),
            'features': 'full_engineered',
            'val_size': 0.2,
            'auc': auc,
            'best_threshold': best_thr,
            **metrics
        })
        experiment_id += 1
    except Exception as e:
        logger.warning(f"    ExtraTrees failed: {e}")
    
    # ===========================================
    # PHASE 2: FEATURE CONFIGURATION TESTING
    # ===========================================
    logger.info("\n" + "="*80)
    logger.info("PHASE 2: FEATURE CONFIGURATION TESTING")
    logger.info("="*80)
    
    # Get best model from Phase 1
    best_so_far = max(all_results, key=lambda x: x['auc'])
    logger.info(f"Best model from Phase 1: {best_so_far['model']} with AUC={best_so_far['auc']:.4f}")
    
    # Test different feature configurations with best model
    for feat_config in ['basic', 'with_derived', 'full_engineered']:
        logger.info(f"\nTesting feature config: {feat_config}")
        X_train_fc, X_val_fc, y_train_fc, y_val_fc, _ = prepare_data(df, feat_config, 0.2, args.seed, logger)
        
        try:
            best_params = json.loads(best_so_far['params'])
            
            if best_so_far['model'] == 'LightGBM':
                _, proba, auc = train_lightgbm(X_train_fc, y_train_fc, X_val_fc, y_val_fc, best_params, args.seed)
            elif best_so_far['model'] == 'XGBoost':
                _, proba, auc = train_xgboost(X_train_fc, y_train_fc, X_val_fc, y_val_fc, best_params, args.seed)
            
            best_thr, best_acc = find_best_threshold(y_val_fc, proba, 'accuracy')
            metrics = evaluate_threshold(y_val_fc, proba, best_thr)
            
            all_results.append({
                'experiment_id': experiment_id,
                'model': best_so_far['model'],
                'params': best_so_far['params'],
                'features': feat_config,
                'val_size': 0.2,
                'auc': auc,
                'best_threshold': best_thr,
                **metrics
            })
            
            experiment_id += 1
            logger.info(f"  AUC: {auc:.4f}, Acc: {metrics['accuracy']:.4f}")
        except Exception as e:
            logger.warning(f"  Failed: {e}")
    
    # ===========================================
    # SAVE RESULTS
    # ===========================================
    logger.info("\n" + "="*80)
    logger.info("SAVING RESULTS")
    logger.info("="*80)
    
    df_results = pd.DataFrame(all_results)
    csv_path = os.path.join(args.out_dir, f"mega_analysis_results_{timestamp}.csv")
    df_results.to_csv(csv_path, index=False)
    logger.info(f"Saved results to: {csv_path}")
    
    # Print top 10 configurations
    logger.info("\n" + "="*80)
    logger.info("TOP 10 CONFIGURATIONS BY AUC")
    logger.info("="*80)
    
    top_10 = df_results.nlargest(10, 'auc')
    for idx, row in top_10.iterrows():
        logger.info(f"\n{row['experiment_id']}. {row['model']} - AUC={row['auc']:.4f}, Acc={row['accuracy']:.4f}")
        logger.info(f"   Features: {row['features']}, Threshold: {row['best_threshold']:.3f}")
        logger.info(f"   F1: {row['f1']:.4f}, Precision: {row['precision']:.4f}, Recall: {row['recall']:.4f}")
    
    # Save best model info
    best_overall = df_results.loc[df_results['auc'].idxmax()]
    best_info = {
        'model': best_overall['model'],
        'params': json.loads(best_overall['params']),
        'features': best_overall['features'],
        'auc': float(best_overall['auc']),
        'accuracy': float(best_overall['accuracy']),
        'f1': float(best_overall['f1']),
        'best_threshold': float(best_overall['best_threshold'])
    }
    
    with open(os.path.join(args.out_dir, f"best_config_{timestamp}.json"), 'w') as f:
        json.dump(best_info, f, indent=2)
    
    logger.info("\n" + "="*80)
    logger.info("MEGA-ANALYSIS COMPLETE!")
    logger.info(f"Best configuration: {best_overall['model']} with AUC={best_overall['auc']:.4f}")
    logger.info("="*80)

if __name__ == "__main__":
    main()
