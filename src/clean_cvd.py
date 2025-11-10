# -*- coding: utf-8 -*-
"""


Usage:
python clean_cvd.py --input "D:/Project/Heart diseae/dataset/1/cardio_train.csv" ^
                    --output "D:/Project/Heart diseae/dataset/1/cardio_clean.csv" ^
                    --report "D:/Project/Heart diseae/dataset/1/clean_report.txt"

Author: Ghasem
https://github.com/GhIrani33?tab=repositories

"""

import argparse, os, json
import numpy as np, pandas as pd

def robust_read_csv(path): return pd.read_csv(path, sep=';')
def to_years(days): return np.floor(days / 365.25).astype(int)

# Clinical ranges
R = {
    "ap_hi": (80, 240),
    "ap_lo": (40, 140),
    "height": (120, 230),   # cm
    "weight": (30.0, 200.0),# kg
    "bmi": (10.0, 60.0),    # kg/m^2
    "age_years": (18, 100)
}

def describe_series(s):
    return {
        "count": int(s.shape[0]),
        "nulls": int(s.isna().sum()),
        "min": float(np.nanmin(s)) if s.size else None,
        "p1": float(np.nanpercentile(s, 1)) if s.size else None,
        "p50": float(np.nanpercentile(s, 50)) if s.size else None,
        "p99": float(np.nanpercentile(s, 99)) if s.size else None,
        "max": float(np.nanmax(s)) if s.size else None,
        "dtype": str(s.dtype)
    }

def profile(df):
    nums = ['age','height','weight','ap_hi','ap_lo','cholesterol','gluc','smoke','alco','active','cardio']
    prof = {}
    for c in df.columns:
        try:
            if pd.api.types.is_numeric_dtype(df[c]):
                prof[c] = describe_series(df[c])
            else:
                vc = df[c].value_counts(dropna=False).to_dict()
                prof[c] = {"dtype": str(df[c].dtype), "unique": int(df[c].nunique()), "value_counts": {str(k): int(v) for k,v in vc.items()}}
        except Exception:
            prof[c] = {"dtype": str(df[c].dtype), "error": "profile_failed"}
    return prof

def clean_dataframe(df):
    report = {"rows_in": int(df.shape[0]), "rules": {}, "caps": {}, "ranges_before": {}, "ranges_after": {}}

    # ranges before
    for c in ['age','age_years','height','weight','ap_hi','ap_lo']:
        if c in df.columns:
            report["ranges_before"][c] = describe_series(df[c])

    # Age to years
    df['age_years'] = to_years(df['age'])

    # BP swap if ap_hi < ap_lo
    swap_mask = df['ap_hi'] < df['ap_lo']
    report["rules"]["bp_swapped"] = int(swap_mask.sum())
    df.loc[swap_mask, ['ap_hi','ap_lo']] = df.loc[swap_mask, ['ap_lo','ap_hi']].to_numpy()

    # Cap BP
    df['ap_hi'] = df['ap_hi'].clip(R['ap_hi'][0], R['ap_hi'][1])
    df['ap_lo'] = df['ap_lo'].clip(R['ap_lo'][0], R['ap_lo'][1])

    # Pulse pressure
    df['pulse_pressure'] = df['ap_hi'] - df['ap_lo']

    # Anthropometrics -> BMI
    h_m = df['height'] / 100.0
    bmi = (df['weight'] / (h_m**2)).replace([np.inf, -np.inf], np.nan)
    df['BMI'] = bmi

    # Rule masks
    m_age = (df['age_years'] >= R['age_years'][0]) & (df['age_years'] <= R['age_years'][1])
    m_pp = df['pulse_pressure'] > 0
    m_height = (df['height'] >= R['height'][0]) & (df['height'] <= R['height'][1])
    m_weight = (df['weight'] >= R['weight'][0]) & (df['weight'] <= R['weight'][1])
    m_bmi = (df['BMI'] >= R['bmi'][0]) & (df['BMI'] <= R['bmi'][1])
    m_gender = df['gender'].isin([1,2])
    m_chol = df['cholesterol'].isin([1,2,3])
    m_gluc = df['gluc'].isin([1,2,3])
    m_bin = df['smoke'].isin([0,1]) & df['alco'].isin([0,1]) & df['active'].isin([0,1])

    # Count drops per rule (not mutually exclusive)
    drops = {
        "age_out": int((~m_age).sum()),
        "pp_nonpositive": int((~m_pp).sum()),
        "height_out": int((~m_height).sum()),
        "weight_out": int((~m_weight).sum()),
        "bmi_out": int((~m_bmi).sum()),
        "gender_bad": int((~m_gender).sum()),
        "chol_bad": int((~m_chol).sum()),
        "gluc_bad": int((~m_gluc).sum()),
        "bin_bad": int((~m_bin).sum()),
    }
    report["rules"].update(drops)

    # Final mask (AND of all)
    mask = m_age & m_pp & m_height & m_weight & m_bmi & m_gender & m_chol & m_gluc & m_bin
    report["rows_drop"] = int((~mask).sum())
    dfc = df.loc[mask].copy()

    # Remove id & raw age
    if 'id' in dfc.columns: dfc = dfc.drop(columns=['id'])
    if 'age' in dfc.columns: dfc = dfc.drop(columns=['age'])

    # ranges after
    for c in ['age_years','height','weight','ap_hi','ap_lo','pulse_pressure','BMI']:
        if c in dfc.columns:
            report["ranges_after"][c] = describe_series(dfc[c])

    # target distribution before/after
    if 'cardio' in df.columns:
        report["target_before"] = {str(k): int(v) for k,v in df['cardio'].value_counts().to_dict().items()}
    if 'cardio' in dfc.columns:
        report["target_after"] = {str(k): int(v) for k,v in dfc['cardio'].value_counts().to_dict().items()}

    report["rows_out"] = int(dfc.shape[0])

    # column order
    cols = ['gender','height','weight','ap_hi','ap_lo','pulse_pressure','cholesterol','gluc','smoke','alco','active','age_years','BMI','cardio']
    dfc = dfc[[c for c in cols if c in dfc.columns]]

    return dfc, report

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--report', required=True)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df = robust_read_csv(args.input)
    dfc, rep = clean_dataframe(df)

    dfc.to_csv(args.output, index=False)
    with open(args.report, 'w', encoding='utf-8') as f:
        f.write(json.dumps(rep, indent=2, ensure_ascii=False))

    print("CLEAN DONE.")
    print(f"- Input rows : {rep['rows_in']}")
    print(f"- Output rows: {rep['rows_out']}")
    print(f"- Dropped    : {rep['rows_drop']}")

if __name__ == "__main__":
    main()
