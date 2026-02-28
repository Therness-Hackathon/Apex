#!/usr/bin/env python3
"""
Apex Weld Quality – Phase 2 Pipeline
======================================
Run:  python phase2.py

Pipeline steps:
  1. Ingest data  (reuses Phase 1 pipeline)
  2. Build features + splits + datasets
  3. Train binary classifier  (or load existing checkpoint)
  4. Calibrate probabilities  (temperature scaling on val set)
  5. Select decision threshold  (on val set, fixed for test)
  6. Generate predictions_binary.csv  (val + test)
  7. Full evaluation report  (val + test, plots, error breakdown)
  8. Export all Phase 2 artefacts
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# ── project imports ──────────────────────────────────────────
from src.config import (
    GOOD_WELD_DIR,
    DEFECT_WELD_DIR,
    LABELS_CSV,
    SPLIT_DIR,
    OUTPUT_DIR,
    MODEL_DIR,
    DASHBOARD_DIR,
    LABEL_COL,
    SAMPLE_ID_COL,
    LABEL_MAP,
    SENSOR_COLUMNS,
    FIXED_SEQ_LEN,
    BATCH_SIZE,
    NUM_EPOCHS,
)
from src.data_ingestion import ingest, save_labels_csv
from src.feature_engineering import build_feature_table
from src.splitter import group_split, save_split, load_split
from src.dataset import WeldDataset, compute_normalize_stats
from src.trainer import (
    train_model,
    evaluate,
    predict,
    save_model,
    compute_class_weights,
    WeldClassifier,
    _get_device,
)
from src.calibration import fit_temperature, predict_calibrated, TemperatureScaler
from src.evaluation import (
    compute_binary_metrics,
    threshold_sweep,
    select_threshold,
    error_breakdown,
    full_evaluation_report,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("phase2")

# ── Paths for Phase 2 artefacts ──────────────────────────────
TEMP_PATH      = MODEL_DIR / "temperature.json"
THRESHOLD_PATH = MODEL_DIR / "threshold.json"
PRED_VAL_PATH  = OUTPUT_DIR / "predictions_binary_val.csv"
PRED_TEST_PATH = OUTPUT_DIR / "predictions_binary.csv"
REPORT_PATH    = OUTPUT_DIR / "phase2_report.txt"


def main() -> None:
    logger.info("Phase 2 pipeline – starting …")
    device = _get_device()

    # ─────────────────────────────────────────────────────────
    # 1. Ingest
    # ─────────────────────────────────────────────────────────
    logger.info("── Step 1: Data ingestion ──")
    manifest, sensor_data = ingest(GOOD_WELD_DIR, DEFECT_WELD_DIR)
    save_labels_csv(manifest, LABELS_CSV)
    n_good = (manifest[LABEL_COL] == 0).sum()
    n_defect = (manifest[LABEL_COL] == 1).sum()
    print(f"\n✓ Ingested {len(manifest)} runs  ({n_good} good, {n_defect} defect)")

    # ─────────────────────────────────────────────────────────
    # 2. Features + split + datasets
    # ─────────────────────────────────────────────────────────
    logger.info("── Step 2: Features, split, datasets ──")
    feature_df = build_feature_table(manifest, sensor_data)
    print(f"✓ Feature table: {feature_df.shape[0]} × {feature_df.shape[1]}")

    split_map = group_split(manifest)
    save_split(split_map)

    norm_stats = compute_normalize_stats(sensor_data, split_map["train"])

    # Save norm stats
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_DIR / "normalize_stats.json", "w") as f:
        json.dump({k: v.tolist() for k, v in norm_stats.items()}, f, indent=2)

    datasets = {}
    loaders = {}
    for split_name, ids in split_map.items():
        ds = WeldDataset(
            manifest=manifest,
            sensor_data=sensor_data,
            sample_ids=ids,
            feature_df=feature_df,
            normalize_stats=norm_stats,
        )
        datasets[split_name] = ds
        loaders[split_name] = DataLoader(
            ds, batch_size=BATCH_SIZE,
            shuffle=(split_name == "train"),
            num_workers=0, pin_memory=True, drop_last=False,
        )

    n_features = datasets["train"][0]["features"].shape[0]
    print(f"✓ Datasets built  (train={len(datasets['train'])}, "
          f"val={len(datasets['val'])}, test={len(datasets['test'])})")

    # ─────────────────────────────────────────────────────────
    # 3. Train classifier
    # ─────────────────────────────────────────────────────────
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / "weld_classifier.pt"

    if model_path.exists():
        logger.info("── Step 3: Loading existing model checkpoint ──")
        n_channels = len(SENSOR_COLUMNS)
        model = WeldClassifier(n_channels, n_features).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        print(f"✓ Model loaded from {model_path}")
    else:
        logger.info("── Step 3: Training binary classifier ──")
        train_manifest = manifest[manifest[SAMPLE_ID_COL].isin(split_map["train"])]
        class_weights = compute_class_weights(train_manifest)
        print(f"  Class weights: good={class_weights[0]:.3f}, defect={class_weights[1]:.3f}")

        model, history = train_model(
            train_loader=loaders["train"],
            val_loader=loaders["val"],
            n_features=n_features,
            class_weights=class_weights,
        )
        save_model(model, model_path)

        # Save training history
        hist_df = pd.DataFrame(history)
        hist_df.index.name = "epoch"
        hist_df.to_csv(OUTPUT_DIR / "training_history.csv")
        print(f"✓ Model trained and saved to {model_path}")

    # ─────────────────────────────────────────────────────────
    # 4. Calibrate with temperature scaling (on val set)
    # ─────────────────────────────────────────────────────────
    logger.info("── Step 4: Temperature scaling calibration ──")
    scaler = fit_temperature(model, loaders["val"], device=device)
    T_val = scaler.temperature.item()
    print(f"✓ Learned temperature: T = {T_val:.4f}")

    with open(TEMP_PATH, "w") as f:
        json.dump({"temperature": round(T_val, 6)}, f, indent=2)
    print(f"✓ Temperature saved → {TEMP_PATH}")

    # ─────────────────────────────────────────────────────────
    # 5. Generate calibrated predictions on val + test
    # ─────────────────────────────────────────────────────────
    logger.info("── Step 5: Calibrated predictions ──")

    def _predictions_to_df(results: dict) -> pd.DataFrame:
        rows = []
        for sid, info in results.items():
            rows.append({
                "sample_id": sid,
                "p_defect": round(info["p_defect"], 6),
                "p_good": round(info["p_good"], 6),
                "confidence": round(info["confidence"], 6),
                "label": info["label"],
            })
        return pd.DataFrame(rows)

    val_results = predict_calibrated(scaler, loaders["val"], device=device)
    val_pred_df = _predictions_to_df(val_results)
    print(f"  Val predictions: {len(val_pred_df)} samples")

    test_results = predict_calibrated(scaler, loaders["test"], device=device)
    test_pred_df = _predictions_to_df(test_results)
    print(f"  Test predictions: {len(test_pred_df)} samples")

    # ─────────────────────────────────────────────────────────
    # 6. Select decision threshold (on val set ONLY)
    # ─────────────────────────────────────────────────────────
    logger.info("── Step 6: Threshold selection ──")
    val_labelled = val_pred_df[val_pred_df["label"].notna()].copy()

    threshold = select_threshold(
        np.asarray(val_labelled["label"].values.astype(int)),
        np.asarray(val_labelled["p_defect"].values),
        strategy="f1",
    )
    print(f"✓ Optimal threshold (F1-maximised on val): {threshold:.4f}")

    with open(THRESHOLD_PATH, "w") as f:
        json.dump({"threshold": threshold, "strategy": "f1_on_val"}, f, indent=2)
    print(f"✓ Threshold saved → {THRESHOLD_PATH}")

    # Apply threshold to predictions
    for df in [val_pred_df, test_pred_df]:
        df["pred_defect"] = (df["p_defect"] >= threshold).astype(int)
        df["confidence"] = df.apply(
            lambda r: r["p_defect"] if r["pred_defect"] == 1 else (1.0 - r["p_defect"]),
            axis=1,
        ).round(6)

    # Save predictions
    val_pred_df.to_csv(PRED_VAL_PATH, index=False)
    test_pred_df.to_csv(PRED_TEST_PATH, index=False)
    print(f"✓ Val predictions  → {PRED_VAL_PATH}")
    print(f"✓ Test predictions → {PRED_TEST_PATH}")

    # ─────────────────────────────────────────────────────────
    # 7. Full evaluation (val + test)
    # ─────────────────────────────────────────────────────────
    logger.info("── Step 7: Evaluation reports ──")
    DASHBOARD_DIR.mkdir(parents=True, exist_ok=True)

    report_parts = []

    for split_name, pred_df in [("val", val_pred_df), ("test", test_pred_df)]:
        labelled = pred_df[pred_df["label"].notna()].copy()
        if len(labelled) == 0:
            logger.warning("No labelled samples in %s – skipping evaluation", split_name)
            continue

        result = full_evaluation_report(
            predictions_df=labelled,
            threshold=threshold,
            split_name=split_name,
            save_dir=DASHBOARD_DIR,
        )

        report_parts.append(result["report_text"])
        print(f"\n{result['report_text']}")

        # Print worst errors
        if len(result["fp_df"]) > 0:
            print(f"\n  Worst False Positives ({split_name}):")
            for _, row in result["fp_df"].head(5).iterrows():
                print(f"    {row['sample_id']:30s}  P(defect)={row['p_defect']:.4f}")

        if len(result["fn_df"]) > 0:
            print(f"\n  Worst False Negatives ({split_name}):")
            for _, row in result["fn_df"].head(5).iterrows():
                print(f"    {row['sample_id']:30s}  P(defect)={row['p_defect']:.4f}")

    # ─────────────────────────────────────────────────────────
    # 8. Export – aggregate report
    # ─────────────────────────────────────────────────────────
    logger.info("── Step 8: Export artefacts ──")

    full_report = "\n\n".join(report_parts)
    full_report += f"\n\nDecision Threshold: {threshold:.4f}  (F1-maximised on validation set)"
    full_report += f"\nTemperature:        {T_val:.4f}"
    full_report += "\n\nConfidence Definition:"
    full_report += "\n  confidence = P(predicted_class)"
    full_report += "\n  i.e. P(defect) when pred=defect, P(good)=1-P(defect) when pred=good"
    full_report += "\n  Calibrated via temperature scaling (Guo et al., ICML 2017)"
    full_report += "\n\nThreshold Policy:"
    full_report += f"\n  If P(defect) >= {threshold:.4f} → classify as DEFECT"
    full_report += f"\n  If P(defect) <  {threshold:.4f} → classify as GOOD"
    full_report += "\n  Threshold was fixed before seeing test labels (no test-set leakage)."

    REPORT_PATH.write_text(full_report, encoding="utf-8")
    print(f"\n✓ Full report → {REPORT_PATH}")

    # Summary of all saved artefacts
    print(f"\n{'=' * 60}")
    print("  PHASE 2 – OUTPUT PACKAGE")
    print(f"{'=' * 60}")
    artefacts = [
        ("Model checkpoint", model_path),
        ("Temperature", TEMP_PATH),
        ("Threshold", THRESHOLD_PATH),
        ("Val predictions", PRED_VAL_PATH),
        ("Test predictions", PRED_TEST_PATH),
        ("Evaluation report", REPORT_PATH),
        ("Norm stats", OUTPUT_DIR / "normalize_stats.json"),
        ("Split definition", SPLIT_DIR / "split.json"),
    ]
    for name, path in artefacts:
        status = "✓" if path.exists() else "✗"
        print(f"  {status} {name:25s} → {path}")

    plot_files = sorted(DASHBOARD_DIR.glob("phase2_*.png"))
    print(f"  ✓ Dashboard plots          → {len(plot_files)} PNGs in {DASHBOARD_DIR}")
    for p in plot_files:
        print(f"      {p.name}")

    print(f"\n{'=' * 60}")
    print("  Inference command:")
    print(f"    python src/inference.py --split test")
    print(f"    python src/inference.py --csv <path_to_csv>")
    print(f"{'=' * 60}")

    logger.info("Phase 2 pipeline complete ✓")


if __name__ == "__main__":
    main()
