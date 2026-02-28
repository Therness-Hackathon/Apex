#!/usr/bin/env python3
"""
Generate competition submission CSV.

Schema required:
    sample_id,pred_label_code,p_defect
    sample_0001,02,0.94
    ...
    sample_0090,00,0.08

Run modes
---------
1. From internal test split (default – uses pre-trained model + feature table):
       python generate_submission.py

2. From a competition test manifest that maps sample_XXXX → data folder path:
       python generate_submission.py --manifest path/to/test_data_manifest.csv

   The manifest must have columns:  sample_id, data_dir
   where data_dir points to the folder containing the CSV + images/ sub-folder.

3. Evaluate against ground-truth labels and print metrics alongside generation:
       python generate_submission.py --eval

Output
------
    outputs/submission.csv
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    OUTPUT_DIR, MODEL_DIR, SPLIT_DIR, LABELS_CSV,
    SENSOR_COLUMNS, FIXED_SEQ_LEN, BATCH_SIZE,
)
from src.multiclass import (
    DEFECT_TYPE_TO_CODE,
    predict_type_codes,
    train_multiclass,
    _MC_MODEL_PATH,
    load_multiclass_model,
    _feature_matrix,
)
from src.trainer import WeldClassifier, _get_device
from src.calibration import TemperatureScaler
from src.feature_engineering import (
    extract_sensor_features,
    extract_image_features,
    sensor_to_fixed_tensor,
    build_feature_table,
)
from src.splitter import load_split
from src.dataset import WeldDataset, compute_normalize_stats

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s")
logger = logging.getLogger("generate_submission")

# ── Artefact paths ───────────────────────────────────────────────────────────
MODEL_PATH      = MODEL_DIR / "weld_classifier.pt"
TEMP_PATH       = MODEL_DIR / "temperature.json"
THRESHOLD_PATH  = MODEL_DIR / "threshold.json"
NORM_STATS_PATH = OUTPUT_DIR / "normalize_stats.json"
_MULTIMODAL_CSV = OUTPUT_DIR / "multimodal_features.csv"
FEATURE_TABLE   = _MULTIMODAL_CSV if _MULTIMODAL_CSV.exists() else OUTPUT_DIR / "feature_table.csv"
SUBMISSION_OUT  = OUTPUT_DIR / "submission.csv"


# ── Model helpers ────────────────────────────────────────────────────────────

def _load_binary_model(device, n_features: int = 104):
    model = WeldClassifier(len(SENSOR_COLUMNS), n_features).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    return model


def _get_temperature(model, device) -> TemperatureScaler:
    scaler = TemperatureScaler(model).to(device)
    if TEMP_PATH.exists():
        with open(TEMP_PATH) as f:
            T = json.load(f)["temperature"]
        scaler.temperature.data.fill_(T)
    return scaler


def _get_threshold() -> float:
    if THRESHOLD_PATH.exists():
        with open(THRESHOLD_PATH) as f:
            return float(json.load(f)["threshold"])
    return 0.50


def _get_norm_stats():
    with open(NORM_STATS_PATH) as f:
        raw = json.load(f)
    return {k: np.array(v, dtype=np.float32) for k, v in raw.items()}


# ── Mode A – infer from raw data folders ─────────────────────────────────────

def _infer_from_manifest(manifest_csv: Path) -> pd.DataFrame:
    """Run full inference pipeline on competition test manifest.

    manifest_csv columns: sample_id, data_dir
    """
    mf = pd.read_csv(manifest_csv)
    required = {"sample_id", "data_dir"}
    if not required.issubset(mf.columns):
        raise ValueError(
            f"Manifest must have columns {required}, got {list(mf.columns)}"
        )

    device = _get_device()
    norm_stats = _get_norm_stats()
    threshold = _get_threshold()

    # We'll collect per-sample sensor features for multi-class too
    feat_rows: list[dict] = []
    bin_results: dict[str, dict] = {}

    # ----- Build feature tensors dynamically -----
    # First pass: compute features for all samples without loading model
    all_feats: dict[str, np.ndarray] = {}
    for _, row in mf.iterrows():
        sid   = str(row["sample_id"])
        ddir  = Path(row["data_dir"])
        csv_files = list(ddir.glob("*.csv"))
        if not csv_files:
            logger.warning("No CSV found in %s – skipping", ddir)
            continue
        raw_df = pd.read_csv(csv_files[0])
        raw_df.columns = raw_df.columns.str.strip()
        feats = extract_sensor_features(raw_df)
        imgs_dir = ddir / "images"
        img_paths = (
            sorted(imgs_dir.glob("*.jpg")) + sorted(imgs_dir.glob("*.png"))
            if imgs_dir.exists() else []
        )
        feats.update(extract_image_features(img_paths))
        feat_vals = np.nan_to_num(
            np.array(list(feats.values()), dtype=np.float32), nan=0.0
        )
        all_feats[sid] = feat_vals
        # Keep a row for the temporary feature DataFrame used by multi-class
        feat_row = {"sample_id": sid, **feats}
        feat_rows.append(feat_row)

    feat_df = pd.DataFrame(feat_rows)

    # ----- Binary inference -----
    n_features = len(next(iter(all_feats.values()))) if all_feats else 104
    model   = _load_binary_model(device, n_features)
    scaler  = _get_temperature(model, device)
    mu, sd  = norm_stats["mean"], norm_stats["std"].copy()
    sd[sd == 0] = 1.0

    for sid, feat_vals in all_feats.items():
        # Need sequence too – reconstruct from raw CSV
        ddir = Path(mf.loc[mf["sample_id"] == sid, "data_dir"].values[0])
        csv_files = list(ddir.glob("*.csv"))
        raw_df = pd.read_csv(csv_files[0])
        raw_df.columns = raw_df.columns.str.strip()
        seq = sensor_to_fixed_tensor(raw_df, FIXED_SEQ_LEN)
        seq = (seq - mu) / sd
        sensor_t = torch.from_numpy(seq).unsqueeze(0).to(device)
        feat_t   = torch.from_numpy(feat_vals).unsqueeze(0).to(device)

        with torch.no_grad():
            probs = scaler.calibrated_probs(sensor_t, feat_t)
        p_defect = float(probs[0, 1].item())
        pred     = 1 if p_defect >= threshold else 0
        bin_results[sid] = {"pred_defect": pred, "p_defect": p_defect}

    # ----- Multi-class type inference -----
    binary_preds = {sid: v["pred_defect"] for sid, v in bin_results.items()}
    type_codes   = predict_type_codes(
        list(all_feats.keys()), feat_df, binary_preds
    )

    # ----- Assemble submission -----
    rows = []
    for _, row in mf.iterrows():
        sid = str(row["sample_id"])
        if sid not in bin_results:
            rows.append({"sample_id": sid, "pred_label_code": "00", "p_defect": 0.5})
            continue
        p_defect = round(bin_results[sid]["p_defect"], 6)
        code     = type_codes.get(sid, "00")
        rows.append({"sample_id": sid, "pred_label_code": code, "p_defect": p_defect})

    return pd.DataFrame(rows)


# ── Mode B – use pre-computed internal test predictions ──────────────────────

def _infer_from_internal_test(n_samples: int = 90) -> pd.DataFrame:
    """Generate submission from our internal test split.

    Reads pre-computed binary predictions from outputs/test_predictions.csv
    and runs the multi-class classifier on the pre-computed feature table.

    Samples are stratified (good / defect proportional) to be representative.
    sample_id is remapped to competition format: sample_0001 … sample_NNNN.
    """
    # ----- Load binaries -----
    test_pred_csv = OUTPUT_DIR / "test_predictions.csv"
    if not test_pred_csv.exists():
        raise FileNotFoundError(
            f"Binary predictions not found at {test_pred_csv}. "
            "Run phase2.py first."
        )
    preds = pd.read_csv(test_pred_csv)
    # Column rename compatibility
    if "prob_defect" in preds.columns and "p_defect" not in preds.columns:
        preds = preds.rename(columns={"prob_defect": "p_defect"})
    if "pred" in preds.columns and "pred_defect" not in preds.columns:
        preds = preds.rename(columns={"pred": "pred_defect"})

    # Stratified sample: preserve good/defect ratio
    if n_samples < len(preds):
        from sklearn.model_selection import train_test_split
        # Use ground-truth label from labels.csv for stratification
        labels = pd.read_csv(LABELS_CSV)[["sample_id", "label", "defect_type"]]
        # Drop any pre-existing label column from preds to avoid merge conflict
        preds_clean = preds.drop(columns=[c for c in ["label", "true_label"] if c in preds.columns])
        preds_with_label = preds_clean.merge(labels, on="sample_id", how="left")
        preds_with_label["label"] = preds_with_label["label"].fillna(0).astype(int)

        try:
            _, preds_sub = train_test_split(
                preds_with_label,
                test_size=n_samples,
                stratify=preds_with_label["label"],
                random_state=42,
            )
            preds = preds_sub.reset_index(drop=True)
        except Exception:
            # Fall back to head() if stratify fails
            preds = preds.head(n_samples).reset_index(drop=True)
    else:
        preds = preds.reset_index(drop=True)

    logger.info("Using %d internal test samples for submission.", len(preds))

    # ----- Multi-class type inference -----
    feat_df = pd.read_csv(FEATURE_TABLE)
    binary_preds  = dict(zip(preds["sample_id"], preds["pred_defect"].astype(int)))
    weld_ids      = preds["sample_id"].tolist()
    type_codes    = predict_type_codes(weld_ids, feat_df, binary_preds)

    # ----- Assemble with competition sample IDs -----
    rows = []
    for i, (_, row) in enumerate(preds.iterrows()):
        comp_id  = f"sample_{i + 1:04d}"
        weld_id  = row["sample_id"]
        p_defect = round(float(row["p_defect"]), 6)
        code     = type_codes.get(weld_id, "00")
        rows.append({
            "sample_id":      comp_id,
            "pred_label_code": code,
            "p_defect":        p_defect,
            # Keep original ID for reference (not in final submission CSV)
            "_weld_id": weld_id,
        })

    df = pd.DataFrame(rows)
    return df


# ── Validation helpers ───────────────────────────────────────────────────────

def _validate_submission(df: pd.DataFrame) -> None:
    """Assert submission schema is correct; raise on violations."""
    required_cols = {"sample_id", "pred_label_code", "p_defect"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    allowed_codes = {"00", "01", "02", "06", "07", "08", "11"}
    bad_codes = set(df["pred_label_code"].unique()) - allowed_codes
    if bad_codes:
        raise ValueError(f"Invalid pred_label_code values: {bad_codes}")

    out_of_range = df[(df["p_defect"] < 0) | (df["p_defect"] > 1)]
    if len(out_of_range):
        raise ValueError(
            f"{len(out_of_range)} rows have p_defect outside [0, 1]"
        )

    dups = df[df["sample_id"].duplicated()]
    if len(dups):
        raise ValueError(f"Duplicate sample_ids: {dups['sample_id'].tolist()}")

    logger.info("Submission validation passed.")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate competition submission CSV")
    parser.add_argument(
        "--manifest", type=Path, default=None,
        help="Path to test_data_manifest.csv (sample_id, data_dir). "
             "If omitted, uses internal test split."
    )
    parser.add_argument(
        "--n-samples", type=int, default=90,
        help="Number of test samples (default: 90, ignored when --manifest given)."
    )
    parser.add_argument(
        "--output", type=Path, default=SUBMISSION_OUT,
        help=f"Output CSV path (default: {SUBMISSION_OUT})"
    )
    parser.add_argument(
        "--train-mc", action="store_true",
        help="Force re-training of the multi-class classifier."
    )
    parser.add_argument(
        "--eval", action="store_true",
        help="After generation, run evaluate_submission.py on the result."
    )
    args = parser.parse_args()

    # ----- Ensure multi-class model exists -----
    if args.train_mc or not _MC_MODEL_PATH.exists():
        logger.info("Training multi-class classifier …")
        train_multiclass(
            split_json=SPLIT_DIR / "split.json",
        )

    # ----- Generate submission -----
    if args.manifest is not None:
        logger.info("Mode: competition manifest (%s)", args.manifest)
        sub = _infer_from_manifest(args.manifest)
    else:
        logger.info("Mode: internal test split (%d samples)", args.n_samples)
        sub = _infer_from_internal_test(n_samples=args.n_samples)

    # ----- Validate -----
    pub_sub = sub[["sample_id", "pred_label_code", "p_defect"]].copy()
    # Ensure codes are string-typed so pandas writes them with leading zeros
    pub_sub["pred_label_code"] = pub_sub["pred_label_code"].astype(str).str.zfill(2)
    _validate_submission(pub_sub)

    # ----- Save internal manifest (for evaluation with ground truth) -----
    if "_weld_id" in sub.columns:
        manifest_path = args.output.with_name(args.output.stem + "_manifest.csv")
        sub[["sample_id", "_weld_id", "pred_label_code", "p_defect"]].assign(
            pred_label_code=pub_sub["pred_label_code"]
        ).to_csv(manifest_path, index=False)
        logger.info("Internal manifest saved → %s", manifest_path)

    # ----- Save -----
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pub_sub.to_csv(args.output, index=False)
    logger.info("Submission saved → %s  (%d rows)", args.output, len(pub_sub))

    print("\nFirst 10 rows:")
    print(pub_sub.head(10).to_string(index=False))

    # ----- Distribution summary -----
    print("\nPred label code distribution:")
    print(pub_sub["pred_label_code"].value_counts().to_string())
    print(f"\np_defect stats: mean={pub_sub['p_defect'].mean():.3f}  "
          f"min={pub_sub['p_defect'].min():.3f}  max={pub_sub['p_defect'].max():.3f}")

    if args.eval:
        import subprocess
        result = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "evaluate_submission.py"),
             "--submission", str(args.output)],
            capture_output=False
        )

    return pub_sub


if __name__ == "__main__":
    main()
