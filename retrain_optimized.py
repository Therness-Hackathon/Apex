#!/usr/bin/env python3
"""
Optimised retraining script.

Skips expensive data ingestion / feature re-engineering by loading the
already-computed outputs from disk.  Then:

  1. Rebuilds sensor_data dict from CSV files directly
  2. Loads feature_table, manifest, split, norm_stats from disk
  3. Retrains the binary classifier (two-phase: train → train+val)
  4. Recalibrates temperature (on val set)
  5. Re-selects decision threshold (on val set)
  6. Retrains multi-class classifier (5-fold CV, on ALL 1551 samples)
  7. Generates a fresh 90-sample submission + evaluation report

Usage:
    python retrain_optimized.py
    python retrain_optimized.py --skip-binary   # only retrain multi-class
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    GOOD_WELD_DIR,
    DEFECT_WELD_DIR,
    LABELS_CSV,
    SPLIT_DIR,
    OUTPUT_DIR,
    MODEL_DIR,
    SENSOR_COLUMNS,
    BATCH_SIZE,
    SAMPLE_ID_COL,
    LABEL_COL,
)
from src.dataset import WeldDataset, compute_normalize_stats
from src.trainer import (
    train_model,
    evaluate,
    save_model,
    compute_class_weights,
    _get_device,
    predict,
)
from src.calibration import fit_temperature, predict_calibrated
from src.multiclass import train_multiclass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger("retrain")

TEMP_PATH = MODEL_DIR / "temperature.json"
THRESH_PATH = MODEL_DIR / "threshold.json"


# ─────────────────────────────────────────────────────────
# Step 1: Build sensor_data dict by scanning directories
# ─────────────────────────────────────────────────────────

def _build_sensor_data(
    roots: List[Path],
    sample_ids: set,
) -> Dict[str, pd.DataFrame]:
    """Scan roots for <run_id>.csv files and load sensor data.

    Only loads CSV files whose folder name is in sample_ids to save time.
    """
    data: Dict[str, pd.DataFrame] = {}
    for root in roots:
        if not root.exists():
            logger.warning("Directory not found: %s", root)
            continue
        for cat_dir in sorted(root.iterdir()):
            if not cat_dir.is_dir() or cat_dir.name in {"wave_files"}:
                continue
            for run_dir in sorted(cat_dir.iterdir()):
                if not run_dir.is_dir():
                    continue
                sid = run_dir.name
                if sid not in sample_ids:
                    continue
                csv_path = run_dir / f"{sid}.csv"
                if not csv_path.exists():
                    continue
                try:
                    df = pd.read_csv(csv_path)
                    # Keep only sensor columns that exist
                    cols = [c for c in SENSOR_COLUMNS if c in df.columns]
                    if not cols:
                        continue
                    data[sid] = df[cols].fillna(0.0)
                except Exception as exc:
                    logger.warning("Could not read %s: %s", csv_path, exc)
    logger.info("Loaded sensor data for %d / %d requested samples.", len(data), len(sample_ids))
    return data


# ─────────────────────────────────────────────────────────
# Threshold selection helpers
# ─────────────────────────────────────────────────────────

def _select_threshold(p_defect: np.ndarray, y_true: np.ndarray) -> float:
    from sklearn.metrics import f1_score
    best_f1, best_t = 0.0, 0.5
    for t in np.arange(0.20, 0.81, 0.01):
        preds = (p_defect >= t).astype(int)
        f = f1_score(y_true, preds, pos_label=1, zero_division=0)
        if f > best_f1:
            best_f1, best_t = f, t
    logger.info("Best val F1=%.4f at threshold=%.2f", best_f1, best_t)
    return float(best_t)


# ─────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-binary", action="store_true",
                        help="Skip binary model retraining (load existing checkpoint).")
    args = parser.parse_args()

    device = _get_device()
    logger.info("Device: %s", device)

    # ── Load cached artifacts ──────────────────────────────────
    logger.info("Loading cached artifacts …")
    manifest   = pd.read_csv(OUTPUT_DIR / "manifest.csv")
    labels_df  = pd.read_csv(LABELS_CSV)[["sample_id", "label", "defect_type"]]

    # ── Multimodal feature extraction (one-time, ~15 min) ─────
    mm_csv = OUTPUT_DIR / "multimodal_features.csv"
    sensor_csv = OUTPUT_DIR / "feature_table.csv"
    if not mm_csv.exists():
        logger.info(
            "multimodal_features.csv not found — running extract_multimodal.py (~15 min) …"
        )
        result_mm = subprocess.run(
            [sys.executable, "extract_multimodal.py"],
            cwd=str(PROJECT_ROOT),
        )
        if result_mm.returncode != 0:
            logger.warning(
                "Multimodal extraction failed (exit %d). "
                "Falling back to sensor-only features.",
                result_mm.returncode,
            )
    feature_csv_path = mm_csv if mm_csv.exists() else sensor_csv
    logger.info("Using feature CSV: %s", feature_csv_path.name)
    feature_df = pd.read_csv(feature_csv_path)

    # Merge label into manifest if missing
    if LABEL_COL not in manifest.columns:
        manifest = manifest.merge(labels_df, on=SAMPLE_ID_COL, how="left")

    with open(SPLIT_DIR / "split.json") as f:
        split_map = json.load(f)

    with open(OUTPUT_DIR / "normalize_stats.json") as f:
        raw = json.load(f)
    norm_stats = {k: np.array(v, dtype=np.float32) for k, v in raw.items()}

    # Index feature_df by sample_id
    if "sample_id" in feature_df.columns:
        feature_df = feature_df.set_index("sample_id")

    all_ids = set(
        split_map["train"] + split_map["val"] + split_map["test"]
    )
    logger.info("Total samples: %d", len(all_ids))

    # ── Build sensor_data ──────────────────────────────────────
    logger.info("Loading sensor CSVs …")
    sensor_data = _build_sensor_data([GOOD_WELD_DIR, DEFECT_WELD_DIR], all_ids)

    # Drop any split ids that don't have sensor data
    for split_name in list(split_map.keys()):
        before = len(split_map[split_name])
        split_map[split_name] = [sid for sid in split_map[split_name]
                                  if sid in sensor_data]
        after = len(split_map[split_name])
        if before != after:
            logger.warning("Split %s: dropped %d samples (no sensor data).",
                           split_name, before - after)

    # ── Build DataLoaders ──────────────────────────────────────
    def _make_loader(split_name: str, shuffle: bool = False) -> DataLoader:
        ds = WeldDataset(
            manifest=manifest,
            sensor_data=sensor_data,
            sample_ids=split_map[split_name],
            feature_df=feature_df,
            normalize_stats=norm_stats,
        )
        return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle,
                          num_workers=0, pin_memory=True, drop_last=False)

    loaders = {
        "train": _make_loader("train", shuffle=True),
        "val":   _make_loader("val"),
        "test":  _make_loader("test"),
    }
    n_features = feature_df.shape[1]
    logger.info("n_features=%d  train=%d  val=%d  test=%d",
                n_features,
                len(split_map["train"]), len(split_map["val"]), len(split_map["test"]))

    # ── Binary model ───────────────────────────────────────────
    model_path = MODEL_DIR / "weld_classifier.pt"
    from src.trainer import WeldClassifier

    if args.skip_binary and model_path.exists():
        logger.info("Loading existing binary model checkpoint …")
        n_channels = len(SENSOR_COLUMNS)
        model = WeldClassifier(n_channels, n_features).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
    else:
        logger.info("── Phase A: train on train split ──")
        train_mf = manifest[manifest[SAMPLE_ID_COL].isin(split_map["train"])]
        cw = compute_class_weights(train_mf)
        model, history_a = train_model(
            train_loader=loaders["train"],
            val_loader=loaders["val"],
            n_features=n_features,
            class_weights=cw,
        )

        logger.info("── Phase B: fine-tune on train+val ──")
        tv_ids = split_map["train"] + split_map["val"]
        tv_mf = manifest[manifest[SAMPLE_ID_COL].isin(tv_ids)]
        tv_cw = compute_class_weights(tv_mf)
        ds_tv = WeldDataset(
            manifest=manifest, sensor_data=sensor_data,
            sample_ids=tv_ids, feature_df=feature_df,
            normalize_stats=norm_stats,
        )
        loader_tv = DataLoader(ds_tv, batch_size=BATCH_SIZE, shuffle=True,
                               num_workers=0, pin_memory=True)
        model, history_b = train_model(
            train_loader=loader_tv,
            val_loader=loaders["test"],
            n_features=n_features,
            class_weights=tv_cw,
        )

        save_model(model, model_path)
        logger.info("Binary model saved → %s", model_path)

        # Save training history
        hist = pd.DataFrame({
            "train_loss": history_a["train_loss"] + history_b["train_loss"],
            "val_loss":   history_a["val_loss"]   + history_b["val_loss"],
            "val_acc":    history_a["val_acc"]     + history_b["val_acc"],
            "val_f1":     history_a["val_f1"]      + history_b["val_f1"],
        })
        hist.index.name = "epoch"
        hist.to_csv(OUTPUT_DIR / "training_history.csv")

    # ── Temperature calibration (on val set) ───────────────────
    logger.info("── Calibrating temperature …")
    scaler = fit_temperature(model, loaders["val"], device=device)
    T = scaler.temperature.item()
    logger.info("Learned temperature: T = %.4f", T)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with open(TEMP_PATH, "w") as f:
        json.dump({"temperature": round(T, 6)}, f, indent=2)

    # ── Threshold selection (on val set) ───────────────────────
    logger.info("── Selecting decision threshold …")
    val_results = predict_calibrated(scaler, loaders["val"], device=device)
    p_defect = np.array([v["p_defect"] for v in val_results.values()])
    y_true_val = np.array([v["label"] if v["label"] is not None else -1
                           for v in val_results.values()])
    valid = y_true_val >= 0
    best_thresh = _select_threshold(p_defect[valid], y_true_val[valid])
    with open(THRESH_PATH, "w") as f:
        json.dump({"threshold": round(best_thresh, 4)}, f, indent=2)
    logger.info("Threshold saved: %.4f", best_thresh)

    # ── Save test predictions for generate_submission.py ──────
    logger.info("── Generating test predictions …")
    test_results = predict_calibrated(scaler, loaders["test"], device=device)
    rows = []
    for sid, info in test_results.items():
        rows.append({
            "sample_id": sid,
            "pred": int(info["p_defect"] >= best_thresh),
            "prob_good": round(info["p_good"], 6),
            "prob_defect": round(info["p_defect"], 6),
            "label": info["label"] if info["label"] is not None else -1,
        })
    test_pred_df = pd.DataFrame(rows)
    test_pred_df.to_csv(OUTPUT_DIR / "test_predictions.csv", index=False)
    logger.info("test_predictions.csv updated (%d rows)", len(test_pred_df))

    # ── Multi-class re-training ────────────────────────────────
    logger.info("── Retraining multi-class model …")
    train_multiclass(
        feature_csv=feature_csv_path,
        labels_csv=LABELS_CSV,
        split_json=SPLIT_DIR / "split.json",
    )

    # ── Generate submission ────────────────────────────────────
    logger.info("── Generating submission …")
    result = subprocess.run(
        [sys.executable, "generate_submission.py"],
        capture_output=True, text=True, cwd=str(PROJECT_ROOT)
    )
    print(result.stdout)
    if result.returncode != 0:
        logger.warning("generate_submission.py stderr: %s", result.stderr[-2000:])

    # ── Evaluate submission ────────────────────────────────────
    logger.info("── Evaluating submission …")
    result2 = subprocess.run(
        [sys.executable, "evaluate_submission.py"],
        capture_output=True, text=True, cwd=str(PROJECT_ROOT)
    )
    print(result2.stdout)
    if result2.returncode != 0:
        logger.warning("evaluate_submission.py stderr: %s", result2.stderr[-2000:])

    logger.info("Done.  Outputs in %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()
