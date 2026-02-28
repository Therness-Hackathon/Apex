#!/usr/bin/env python3
"""Validate submission.csv against competition rules."""
import sys
import pandas as pd

df = pd.read_csv("outputs/submission.csv", dtype={"pred_label_code": str})

errors = []

# 1. Exactly 90 rows
if len(df) != 90:
    errors.append(f"Expected 90 rows, got {len(df)}")

# 2. Required columns
for col in ["sample_id", "pred_label_code", "p_defect"]:
    if col not in df.columns:
        errors.append(f"Missing column: {col}")

if not errors:
    # 3. sample_ids must be sample_0001..sample_0090, in order, no duplicates
    expected = [f"sample_{i:04d}" for i in range(1, 91)]
    if df["sample_id"].tolist() != expected:
        errors.append(f"sample_id mismatch. Got: {df['sample_id'].tolist()[:5]}...")
    if df["sample_id"].nunique() != 90:
        errors.append("Duplicate sample_ids")

    # 4. pred_label_code from allowed set
    allowed = {"00", "01", "02", "04", "06", "11"}
    df["pred_label_code"] = df["pred_label_code"].astype(str)
    bad = set(df["pred_label_code"].unique()) - allowed
    if bad:
        errors.append(f"Disallowed codes: {bad}")

    # 5. p_defect in [0,1]
    if not df["p_defect"].between(0, 1).all():
        out = df[~df["p_defect"].between(0, 1)]
        errors.append(f"p_defect out of [0,1]: {out}")

if errors:
    print("VALIDATION FAILED:")
    for e in errors:
        print(f"  ✗  {e}")
    sys.exit(1)

print("ALL CHECKS PASSED ✓")
print(f"  Rows     : {len(df)}")
print(f"  Columns  : {df.columns.tolist()}")
print(f"  Codes    : {sorted(df['pred_label_code'].unique())}")
print(f"  p_defect : min={df['p_defect'].min():.4f}  max={df['p_defect'].max():.4f}  mean={df['p_defect'].mean():.4f}")
print()
print("Code distribution:")
print(df["pred_label_code"].value_counts().sort_index().to_string())
print()
print(df.to_string(index=False))
