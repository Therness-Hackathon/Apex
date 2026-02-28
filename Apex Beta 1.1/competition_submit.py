#!/usr/bin/env python3
"""
Competition Submission Generator

This script generates a proper submission for the competition by:
1. Extracting audio features from our training data (cached in multimodal_features.csv)
2. Extracting audio features from the competition test data (.flac only)
3. Training audio-only binary + multi-class classifiers on training data
4. Predicting on the 115 competition test samples
5. Producing submission.csv in the required format

The competition test data contains ONLY .flac audio files (no sensor CSVs),
so we must use audio-only features for inference.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import OUTPUT_DIR, LABELS_CSV, GOOD_WELD_DIR, DEFECT_WELD_DIR
from src.audio_features import extract_audio_features

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger("competition_submit")

# ── Paths ─────────────────────────────────────────────────────────────────────
COMPETITION_TEST_DIR = PROJECT_ROOT / "data2" / "test_data-20260228T053817Z-1-001" / "test_data"
SUBMISSION_OUT = OUTPUT_DIR / "submission.csv"
AUDIO_CACHE_TRAIN = OUTPUT_DIR / "audio_features_train.csv"
AUDIO_CACHE_TEST = OUTPUT_DIR / "audio_features_test.csv"

# Correct defect code mapping (matches weld ID suffix convention)
# Allowed submission codes: 00, 01, 02, 04, 06, 11
DEFECT_TYPE_TO_CODE = {
    "good":                  "00",
    "excessive_penetration": "01",
    "burnthrough":           "02",
    "lack_of_fusion":        "04",   # competition uses 04 (not 07)
    "overlap":               "06",
    "excessive_convexity":   "04",   # no 08 slot; fold into 04
    "crater_cracks":         "11",
}
ALLOWED_CODES = {"00", "01", "02", "04", "06", "11"}
SUBMISSION_SAMPLES = [f"sample_{i:04d}" for i in range(1, 91)]  # sample_0001..sample_0090


# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Extract audio features from training data
# ─────────────────────────────────────────────────────────────────────────────

def _discover_training_flacs() -> List[Dict]:
    """Find all .flac files in good_weld/ and defect-weld/."""
    runs = []
    skip = {"wave_files", "__pycache__", ".git"}
    for root in [GOOD_WELD_DIR, DEFECT_WELD_DIR]:
        if not root.exists():
            logger.warning("Training dir missing: %s", root)
            continue
        for cat_dir in sorted(root.iterdir()):
            if not cat_dir.is_dir() or cat_dir.name in skip:
                continue
            for run_dir in sorted(cat_dir.iterdir()):
                if not run_dir.is_dir():
                    continue
                sid = run_dir.name
                flac = run_dir / f"{sid}.flac"
                if flac.exists():
                    runs.append({"sample_id": sid, "flac_path": flac})
    # Dedup
    seen = set()
    unique = []
    for r in runs:
        if r["sample_id"] not in seen:
            seen.add(r["sample_id"])
            unique.append(r)
    return unique


def extract_train_audio(force: bool = False) -> pd.DataFrame:
    """Extract audio features for all training samples. Cached to disk."""
    if AUDIO_CACHE_TRAIN.exists() and not force:
        logger.info("Loading cached training audio features from %s", AUDIO_CACHE_TRAIN)
        return pd.read_csv(AUDIO_CACHE_TRAIN)

    runs = _discover_training_flacs()
    logger.info("Extracting audio features for %d training samples...", len(runs))

    from tqdm import tqdm
    rows = []
    for run in tqdm(runs, desc="Train audio"):
        feats = extract_audio_features(run["flac_path"])
        feats["sample_id"] = run["sample_id"]
        rows.append(feats)

    df = pd.DataFrame(rows)
    ok = df.get("aud__file_ok", pd.Series(dtype=float)).sum()
    logger.info("Audio OK: %d / %d", int(ok), len(df))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(AUDIO_CACHE_TRAIN, index=False)
    logger.info("Saved training audio features -> %s", AUDIO_CACHE_TRAIN)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Step 2: Extract audio features from competition test data
# ─────────────────────────────────────────────────────────────────────────────

def _discover_test_samples() -> List[Dict]:
    """Find all competition test samples (sample_XXXX -> .flac)."""
    samples = []
    if not COMPETITION_TEST_DIR.exists():
        logger.error("Competition test dir not found: %s", COMPETITION_TEST_DIR)
        return samples

    for d in sorted(COMPETITION_TEST_DIR.iterdir()):
        if not d.is_dir() or not d.name.startswith("sample_"):
            continue
        flacs = list(d.glob("*.flac"))
        if flacs:
            weld_id = flacs[0].stem  # e.g. "04-03-23-0010-11"
            samples.append({
                "sample_id": d.name,       # sample_0001
                "weld_id": weld_id,
                "flac_path": flacs[0],
            })
    return samples


def extract_test_audio(force: bool = False) -> pd.DataFrame:
    """Extract audio features for competition test samples. Cached to disk."""
    if AUDIO_CACHE_TEST.exists() and not force:
        logger.info("Loading cached test audio features from %s", AUDIO_CACHE_TEST)
        return pd.read_csv(AUDIO_CACHE_TEST)

    samples = _discover_test_samples()
    logger.info("Extracting audio features for %d test samples...", len(samples))

    from tqdm import tqdm
    rows = []
    for s in tqdm(samples, desc="Test audio"):
        feats = extract_audio_features(s["flac_path"])
        feats["sample_id"] = s["sample_id"]
        feats["weld_id"] = s["weld_id"]
        rows.append(feats)

    df = pd.DataFrame(rows)
    ok = df.get("aud__file_ok", pd.Series(dtype=float)).sum()
    logger.info("Audio OK: %d / %d", int(ok), len(df))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(AUDIO_CACHE_TEST, index=False)
    logger.info("Saved test audio features -> %s", AUDIO_CACHE_TEST)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Step 3: Audio feature columns
# ─────────────────────────────────────────────────────────────────────────────

def _audio_feature_cols(df: pd.DataFrame) -> List[str]:
    """Get audio feature column names (exclude IDs and metadata)."""
    skip = {"sample_id", "weld_id", "aud__file_ok"}
    return [c for c in df.columns if c.startswith("aud__") and c not in skip]


# ─────────────────────────────────────────────────────────────────────────────
# Step 4: Train audio-only classifiers + predict competition test
# ─────────────────────────────────────────────────────────────────────────────

def train_and_predict(
    train_audio: pd.DataFrame,
    test_audio: pd.DataFrame,
) -> pd.DataFrame:
    """Train a single 7-class classifier on audio, predict test samples.

    Uses a unified 7-class approach:
      - All 7 defect types (including good="00") in one model
      - Binary prediction derived from whether predicted code == "00"
      - p_defect = 1 - P(good)
      - Ensemble of LR + SVM for robustness
    """

    # Load labels
    labels = pd.read_csv(LABELS_CSV)[["sample_id", "label", "defect_type"]]

    # Merge labels into training features
    train = train_audio.merge(labels, on="sample_id", how="inner")

    # Remove samples with broken audio
    if "aud__file_ok" in train.columns:
        before = len(train)
        train = train[train["aud__file_ok"] > 0].copy()
        logger.info("Removed %d samples with broken audio", before - len(train))

    logger.info("Training samples with labels: %d", len(train))

    # Map defect_type to competition code
    train["code"] = train["defect_type"].map(DEFECT_TYPE_TO_CODE)
    train = train.dropna(subset=["code"])

    feat_cols = _audio_feature_cols(train)
    logger.info("Audio feature columns: %d", len(feat_cols))

    X_train = train[feat_cols].fillna(0.0).values.astype(np.float32)
    y_codes = np.asarray(train["code"].values, dtype=object)  # 7-class: "00","01","02","06","07","08","11"

    # Prepare test
    for c in feat_cols:
        if c not in test_audio.columns:
            test_audio[c] = 0.0
    X_test = test_audio[feat_cols].fillna(0.0).values.astype(np.float32)

    # ── Class distribution ───────────────────────────────────────────────────
    unique, counts = np.unique(np.asarray(y_codes), return_counts=True)
    for cls, cnt in zip(unique, counts):
        logger.info("  Class %s: %d samples", cls, cnt)

    # ── Feature selection + LR+SVM ensemble ──────────────────────────────────
    logger.info("Training LR(C=7)+SVM(C=2) with top-130 features...")

    # Scale features first
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    # Feature selection via LR coefficient importance
    lr_selector = LogisticRegression(
        C=5.0, max_iter=3000, solver="lbfgs",
        class_weight="balanced", random_state=42,
    )
    lr_selector.fit(X_scaled, y_codes)
    importances = np.abs(lr_selector.coef_).max(axis=0)
    top_k = np.argsort(importances)[::-1]
    N_TOP = 130
    selected = top_k[:N_TOP]
    logger.info("Selected top %d features by LR importance", N_TOP)

    X_tr_sel = X_scaled[:, selected]

    # Build LR + SVM ensemble
    lr = LogisticRegression(
        C=7.0, max_iter=3000, solver="lbfgs",
        class_weight="balanced", random_state=42,
    )
    svm = SVC(
        C=2.0, kernel="rbf", class_weight="balanced",
        random_state=42, probability=True,
    )
    ensemble = VotingClassifier(
        estimators=[("lr", lr), ("svm", svm)],
        voting="soft",
        weights=[2, 1],
    )

    # 5-fold CV
    logger.info("Running 5-fold CV...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    from sklearn.model_selection import cross_val_predict
    cv_preds = cross_val_predict(ensemble, X_tr_sel, np.asarray(y_codes, dtype=object), cv=cv)

    from sklearn.metrics import f1_score as _f1
    cv_mc_f1 = _f1(y_codes, cv_preds, average="macro")
    y_bin = (y_codes != "00").astype(int)
    cv_bin_preds = (cv_preds != "00").astype(int)
    cv_bin_f1 = f1_score(y_bin, cv_bin_preds)
    logger.info("7-class CV Macro-F1: %.4f", cv_mc_f1)
    logger.info("Binary CV F1 (from 7-class): %.4f", cv_bin_f1)

    # Fit on all training data
    logger.info("Fitting final model on all training data...")
    ensemble.fit(X_tr_sel, np.asarray(y_codes, dtype=object))

    # Predict test
    X_test_scaled = scaler.transform(X_test)
    X_te_sel = X_test_scaled[:, selected]
    mc_preds = ensemble.predict(X_te_sel)
    mc_proba = ensemble.predict_proba(X_te_sel)
    classes = ensemble.classes_

    # p_defect = 1 - P(code "00")
    good_idx = list(classes).index("00")
    p_good = mc_proba[:, good_idx]
    p_defect = 1.0 - p_good

    # Zero-pad codes to 2 chars (e.g. "4" -> "04")
    mc_preds_arr = np.asarray(mc_preds, dtype=object)
    mc_preds_str = [c.zfill(2) for c in mc_preds_arr.astype(str)]
    return pd.DataFrame({
        "sample_id": test_audio["sample_id"].values,
        "pred_label_code": mc_preds_str,
        "p_defect": np.round(p_defect, 6),
    })


def validate_submission(sub: pd.DataFrame) -> None:
    """Basic validation checks on the submission."""
    required_cols = {"sample_id", "pred_label_code", "p_defect"}
    assert required_cols.issubset(sub.columns), f"Missing columns: {required_cols - set(sub.columns)}"

    n = len(sub)
    logger.info("Submission: %d samples", n)

    # Check codes
    codes = sub["pred_label_code"].unique()
    logger.info("Predicted codes: %s", sorted(codes))

    # Distribution
    dist = sub["pred_label_code"].value_counts().sort_index()
    logger.info("Distribution:\n%s", dist.to_string())

    # p_defect range
    p = sub["p_defect"]
    logger.info("p_defect: mean=%.3f  min=%.4f  max=%.4f", p.mean(), p.min(), p.max())


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate competition submission")
    parser.add_argument("--force-extract", action="store_true",
                        help="Re-extract audio features even if cached")
    args = parser.parse_args()

    force = args.force_extract

    # Step 1: Extract training audio features
    train_audio = extract_train_audio(force=force)

    # Step 2: Extract competition test audio features
    test_audio = extract_test_audio(force=force)

    logger.info("Train: %d samples, Test: %d samples", len(train_audio), len(test_audio))

    # Step 3: Train and predict
    submission = train_and_predict(train_audio, test_audio)

    # Step 4: Filter to exactly 90 required samples (sample_0001..sample_0090)
    submission = submission[submission["sample_id"].isin(SUBMISSION_SAMPLES)].copy()
    # Ensure correct order and exactly 90 rows
    submission["_order"] = submission["sample_id"].map(
        {s: i for i, s in enumerate(SUBMISSION_SAMPLES)}
    )
    submission = submission.sort_values("_order").drop(columns=["_order"]).reset_index(drop=True)
    assert len(submission) == 90, f"Expected 90 rows, got {len(submission)}"

    # Step 5: Validate codes are in allowed set
    bad_codes = set(submission["pred_label_code"].unique()) - ALLOWED_CODES
    if bad_codes:
        logger.warning("Remapping disallowed codes %s -> 04", bad_codes)
        submission["pred_label_code"] = submission["pred_label_code"].apply(
            lambda c: c if c in ALLOWED_CODES else "04"
        )

    # Step 6: Validate
    validate_submission(submission)

    # Step 7: Save
    submission.to_csv(SUBMISSION_OUT, index=False)
    logger.info("Submission saved -> %s  (%d rows)", SUBMISSION_OUT, len(submission))

    # Print summary
    print(f"\nSubmission: {len(submission)} rows")
    print("\nFirst 10 rows:")
    print(submission.head(10).to_string(index=False))
    print("\nCode distribution:")
    print(submission["pred_label_code"].value_counts().sort_index().to_string())
    print(f"\nSaved -> {SUBMISSION_OUT}")


if __name__ == "__main__":
    main()
