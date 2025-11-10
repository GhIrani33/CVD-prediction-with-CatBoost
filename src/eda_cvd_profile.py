# -*- coding: utf-8 -*-
# EDA & Data Quality Profile for Kaggle CVD (semicolon-delimited)
# Save as: eda_cvd_profile.py
# Usage: python eda_cvd_profile.py --input "D:\Project\Heart diseae\dataset\1\cardio_train.csv" --report "cvd_eda_report.txt"
# Author: Ghasem, https://github.com/GhIrani33?tab=repositories

import argparse
import os
import io
import sys
import time
import math
import json
import numpy as np
import pandas as pd
from collections import Counter
from scipy import stats
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import mutual_info_score
from itertools import combinations

def robust_read_csv(path):
    # Kaggle CVD uses semicolon separator
    return pd.read_csv(path, sep=';')

def dtype_overview(df):
    buf = io.StringIO()
    df.info(buf=buf)
    return buf.getvalue()

def detect_hidden_missing(df):
    # Detect blanks, spaces, special tokens like 'NA','NaN','None','?'
    tokens = {'', ' ', 'NA', 'N/A', 'NaN', 'nan', 'None', '?', 'null', 'NULL'}
    report = []
    for col in df.columns:
        if df[col].dtype == object:
            mask = df[col].astype(str).str.strip().isin(tokens)
            count = mask.sum()
            if count > 0:
                report.append((col, int(count)))
    return report

def summarize_duplicates(df):
    total = len(df)
    dup_rows = df.duplicated().sum()
    # Optional: duplication by feature subsets (id excluded)
    return total, dup_rows

def physiological_checks(df):
    # Clinical plausibility rules based on domain knowledge
    issues = []
    # age (days) should be positive and plausible (e.g., 18-100 years)
    if 'age' in df.columns:
        years = df['age'] / 365.25
        invalid_age = ((years < 18) | (years > 100)).sum()
        issues.append(('age_out_of_range_count', int(invalid_age)))

    # height (cm) plausible range: 120-230
    if 'height' in df.columns:
        invalid_h = ((df['height'] < 120) | (df['height'] > 230)).sum()
        issues.append(('height_out_of_range_count', int(invalid_h)))

    # weight (kg) plausible range: 30-200
    if 'weight' in df.columns:
        invalid_w = ((df['weight'] < 30) | (df['weight'] > 200)).sum()
        issues.append(('weight_out_of_range_count', int(invalid_w)))

    # BP plausibility
    if {'ap_hi','ap_lo'}.issubset(df.columns):
        # Impossible ordering ap_hi < ap_lo
        inv_order = (df['ap_hi'] < df['ap_lo']).sum()
        issues.append(('ap_order_ap_hi_lt_ap_lo', int(inv_order)))
        # Extreme systolic > 250 or < 70
        extreme_sys = ((df['ap_hi'] > 250) | (df['ap_hi'] < 70)).sum()
        issues.append(('ap_hi_extreme', int(extreme_sys)))
        # Extreme diastolic > 150 or < 40
        extreme_dia = ((df['ap_lo'] > 150) | (df['ap_lo'] < 40)).sum()
        issues.append(('ap_lo_extreme', int(extreme_dia)))

    # gender coding plausibility (1,2 expected)
    if 'gender' in df.columns:
        invalid_g = (~df['gender'].isin([1,2])).sum()
        issues.append(('gender_invalid_codes', int(invalid_g)))

    # categorical coding plausibility
    for col, valid in [('cholesterol',[1,2,3]),('gluc',[1,2,3]),('smoke',[0,1]),('alco',[0,1]),('active',[0,1]),('cardio',[0,1])]:
        if col in df.columns:
            bad = (~df[col].isin(valid)).sum()
            issues.append((f'{col}_invalid_codes', int(bad)))
    return issues

def compute_bmi(df):
    if {'height','weight'}.issubset(df.columns):
        h_m = df['height'] / 100.0
        with np.errstate(divide='ignore', invalid='ignore'):
            bmi = df['weight'] / (h_m**2)
        return bmi.replace([np.inf,-np.inf], np.nan)
    return pd.Series([np.nan]*len(df))

def univariate_stats(df):
    desc_num = df.describe(include=[np.number]).T
    # add skewness & kurtosis
    desc_num['skew'] = df[desc_num.index].skew(numeric_only=True)
    desc_num['kurtosis'] = df[desc_num.index].kurtosis(numeric_only=True)
    return desc_num

def categorical_summary(df, cat_cols):
    lines = []
    for c in cat_cols:
        vc = df[c].value_counts(dropna=False).sort_index()
        lines.append(f'-- {c} --')
        lines.append(vc.to_string())
    return '\n'.join(lines)

def correlation_analysis(df, target='cardio'):
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c != target]
    corr = df[num_cols + ([target] if target in df.columns else [])].corr(numeric_only=True)
    return corr

def chi_square_tests(df, target='cardio'):
    # run chi-square for categorical vs target
    out = []
    if target not in df.columns:
        return out
    cat_cols = [c for c in df.columns if (df[c].dtype == 'object') or (df[c].nunique()<=10 and c != target)]
    for c in cat_cols:
        tbl = pd.crosstab(df[c], df[target])
        if tbl.shape[0] > 1 and tbl.shape[1] > 1:
            chi2, p, dof, exp = stats.chi2_contingency(tbl)
            out.append((c, chi2, p, dof))
    return out

def mutual_information(df, target='cardio'):
    # MI for mixed types: discretize continuous for MI with classification
    if target not in df.columns:
        return []
    X = df.drop(columns=[target]).copy()
    y = df[target].values
    X_proc = X.copy()
    # identify continuous columns (numeric with many unique)
    cont_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c]) and X[c].nunique() > 20]
    disc = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='quantile')
    if cont_cols:
        X_proc[cont_cols] = disc.fit_transform(X[cont_cols].fillna(X[cont_cols].median()))
    # fill missing with mode/median
    for c in X_proc.columns:
        if pd.api.types.is_numeric_dtype(X_proc[c]):
            X_proc[c] = X_proc[c].fillna(X_proc[c].median())
        else:
            X_proc[c] = X_proc[c].fillna(X_proc[c].mode().iloc[0])
    mi_scores = []
    for c in X_proc.columns:
        mi = mutual_info_score(X_proc[c], y)
        mi_scores.append((c, mi))
    mi_scores.sort(key=lambda x: x[1], reverse=True)
    return mi_scores

def kolmogorov_smirnov_by_target(df, target='cardio'):
    # For each continuous feature, compare distributions between target classes
    if target not in df.columns:
        return []
    results = []
    cont_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c != target]
    g0 = df[df[target]==0]
    g1 = df[df[target]==1]
    for c in cont_cols:
        x0 = g0[c].dropna().values
        x1 = g1[c].dropna().values
        if len(x0)>100 and len(x1)>100:
            stat, p = stats.ks_2samp(x0, x1)
            results.append((c, stat, p))
    results.sort(key=lambda x: x[1], reverse=True)
    return results

def inconsistency_maps(df):
    # Map impossible physiological combinations count
    issues = {}
    if {'ap_hi','ap_lo'}.issubset(df.columns):
        issues['ap_hi_less_than_ap_lo'] = int((df['ap_hi'] < df['ap_lo']).sum())
        issues['pulse_pressure_negative'] = int(((df['ap_hi'] - df['ap_lo']) <= 0).sum())
        issues['wide_pulse_pressure'] = int(((df['ap_hi'] - df['ap_lo']) >= 100).sum())
    return issues

def main(input_path, report_path):
    t0 = time.time()
    df = robust_read_csv(input_path)

    # Build report text
    rep = []
    rep.append('='*100)
    rep.append('COMPREHENSIVE EDA & DATA QUALITY REPORT – KAGGLE CARDIOVASCULAR DISEASE DATASET')
    rep.append('='*100)
    rep.append(f'File: {input_path}')
    rep.append(f'Date: {pd.Timestamp.now()}')
    rep.append('Source: Kaggle – Sulianova Cardiovascular Disease dataset')
    rep.append('')

    # Basic structure
    rep.append('1) DATASET STRUCTURE')
    rep.append(f'- Shape: {df.shape[0]} rows × {df.shape[1]} columns')
    rep.append('- Columns: ' + ', '.join(df.columns))
    rep.append('')
    rep.append('DataFrame.info():')
    rep.append(dtype_overview(df))
    rep.append('')

    # Hidden missing & explicit missing
    rep.append('2) MISSING & HIDDEN MISSING VALUES')
    miss_counts = df.isna().sum()
    rep.append('Explicit missing counts:')
    rep.append(miss_counts.to_string())
    hidden = detect_hidden_missing(df)
    rep.append('Hidden missing-like tokens in object columns:')
    rep.append(json.dumps(hidden, indent=2))
    rep.append('')

    # Duplicates
    rep.append('3) DUPLICATES')
    total, dups = summarize_duplicates(df)
    rep.append(f'- Total rows: {total}')
    rep.append(f'- Exact duplicate rows: {dups}')
    rep.append('')

    # Physiological plausibility & inconsistencies
    rep.append('4) CLINICAL PLAUSIBILITY & INCONSISTENCY CHECKS')
    issues = physiological_checks(df)
    rep.append('Plausibility issues:')
    rep.append(json.dumps(issues, indent=2))
    rep.append('Inconsistency maps:')
    rep.append(json.dumps(inconsistency_maps(df), indent=2))
    rep.append('')

    # Feature engineering candidates
    rep.append('5) FEATURE ENGINEERING CANDIDATES')
    bmi = compute_bmi(df)
    rep.append('BMI summary (computed):')
    rep.append(bmi.describe().to_string())
    rep.append('Suggested engineered features:')
    rep.append('- BMI and BMI categories (WHO cutoffs)')
    rep.append('- Age in years and age bands (e.g., 18–29, 30–39, ...)')
    rep.append('- BP categories: normal/elevated/stage1/stage2 based on ap_hi/ap_lo')
    rep.append('- Pulse pressure (ap_hi - ap_lo)')
    rep.append('- Interaction: age×BP, BMI×gluc, gender×cholesterol')
    rep.append('')

    # Univariate stats
    rep.append('6) UNIVARIATE STATISTICS (NUMERIC)')
    rep.append(univariate_stats(df).to_string())
    rep.append('')

    # Categorical summaries
    rep.append('7) CATEGORICAL SUMMARIES')
    cat_cols = [c for c in df.columns if (df[c].dtype=='object') or (df[c].nunique()<=10)]
    rep.append(categorical_summary(df, cat_cols))
    rep.append('')

    # Correlation
    rep.append('8) CORRELATION MATRIX (NUMERIC)')
    try:
        corr = correlation_analysis(df, target='cardio')
        rep.append(corr.to_string())
    except Exception as e:
        rep.append(f'Correlation failed: {e}')
    rep.append('')

    # Chi-square tests for categorical vs target
    rep.append('9) CHI-SQUARE TESTS (CATEGORICAL vs TARGET)')
    for c, chi2v, p, dof in chi_square_tests(df, target='cardio'):
        rep.append(f'- {c}: chi2={chi2v:.3f}, p={p:.3e}, dof={dof}')
    rep.append('')

    # MI
    rep.append('10) MUTUAL INFORMATION RANKING (ALL FEATURES vs TARGET)')
    try:
        mi = mutual_information(df, target='cardio')
        for c, v in mi:
            rep.append(f'- {c}: MI={v:.5f}')
    except Exception as e:
        rep.append(f'MI failed: {e}')
    rep.append('')

    # KS-tests by target
    rep.append('11) KS-TESTS BETWEEN TARGET CLASSES (CONTINUOUS FEATURES)')
    for c, stat, p in kolmogorov_smirnov_by_target(df, target='cardio'):
        rep.append(f'- {c}: KS={stat:.4f}, p={p:.3e}')
    rep.append('')

    # Data drift proxies (simple)
    rep.append('12) SIMPLE DRIFT PROXIES (TRAIN/TEST SPLIT SHAPE CHECK)')
    # Not splitting here; placeholder for future integration

    # Write report
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(rep))

    print(f'Written EDA report to: {report_path}')
    print(f'Rows: {df.shape[0]}, Columns: {df.shape[1]}')
    print(f'Elapsed: {time.time()-t0:.2f}s')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Path to cardio_train.csv (semicolon-delimited)')
    parser.add_argument('--report', default='cvd_eda_report.txt', help='Output text report path')
    args = parser.parse_args()
    main(args.input, args.report)
