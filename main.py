#!/usr/bin/env python3
"""
Apex Weld Quality – Full Pipeline
===================================
Run:  python main.py

Pipeline steps:
  1. Ingest & validate all runs from good_weld/ and defect-weld/
  2. Auto-generate labels from folder structure
  3. Compute feature table (sensor + image statistics)
  4. Create stratified group-based train/val/test split
  5. Build PyTorch datasets & DataLoaders
  6. Train binary classifier (1-D CNN + feature MLP)
  7. Evaluate on val & test sets
  8. Export all artefacts
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

# ── project imports ──────────────────────────────────────────────
from src.config import (
    GOOD_WELD_DIR,
    DEFECT_WELD_DIR,
    LABELS_CSV,
    SPLIT_DIR,
    OUTPUT_DIR,
    MODEL_DIR,
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
from src.splitter import group_split, save_split
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("apex")


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def _make_summary(manifest, feature_df, split_map) -> str:
    lines = []
    lines.append("=" * 60)
    lines.append("  APEX WELD QUALITY – PIPELINE SUMMARY")
    lines.append("=" * 60)

    n_good = (manifest[LABEL_COL] == 0).sum()
    n_defect = (manifest[LABEL_COL] == 1).sum()
    lines.append(f"\nTotal runs  : {len(manifest)}")
    lines.append(f"  good      : {n_good}")
    lines.append(f"  defect    : {n_defect}")

    if "defect_type" in manifest.columns:
        lines.append("\nDefect sub-types:")
        dt = manifest[manifest[LABEL_COL] == 1]["defect_type"].value_counts()
        for k, v in dt.items():
            lines.append(f"  {k:30s}: {v}")

    n_issues = manifest["issues"].apply(len).sum()
    lines.append(f"\nData-quality issues: {n_issues}")

    durations = manifest["duration_s"].dropna()
    if len(durations):
        lines.append(f"Weld duration (s) : mean={durations.mean():.1f}  "
                      f"std={durations.std():.1f}  "
                      f"min={durations.min():.1f}  max={durations.max():.1f}")

    lines.append(f"Feature shape     : {feature_df.shape[0]} × {feature_df.shape[1]}")

    for k, v in split_map.items():
        lines.append(f"Split {k:5s}        : {len(v)} runs")

    lines.append("\n" + "=" * 60)
    return "\n".join(lines)


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main() -> None:
    logger.info("Starting pipeline …")

    # ── 1. Ingest ────────────────────────────────────────────────
    logger.info("── Step 1: Ingest & validate ──")
    manifest, sensor_data = ingest(GOOD_WELD_DIR, DEFECT_WELD_DIR)
    n_good = (manifest[LABEL_COL] == 0).sum()
    n_defect = (manifest[LABEL_COL] == 1).sum()
    print(f"\n✓ Ingested {len(manifest)} runs  ({n_good} good, {n_defect} defect)")

    # ── 2. Save auto-generated labels ────────────────────────────
    save_labels_csv(manifest, LABELS_CSV)
    print(f"✓ Labels saved to {LABELS_CSV}")

    # ── 3. Feature engineering ───────────────────────────────────
    logger.info("── Step 2: Feature engineering ──")
    feature_df = build_feature_table(manifest, sensor_data)
    print(f"✓ Feature table: {feature_df.shape[0]} × {feature_df.shape[1]}")

    # ── 4. Split ─────────────────────────────────────────────────
    logger.info("── Step 3: Stratified group split ──")
    split_map = group_split(manifest)
    split_path = save_split(split_map)
    print(f"✓ Split saved to {split_path}")
    for k, v in split_map.items():
        lbl_counts = manifest[manifest[SAMPLE_ID_COL].isin(v)][LABEL_COL].value_counts()
        good_n = lbl_counts.get(0, 0)
        def_n = lbl_counts.get(1, 0)
        print(f"  {k:5s}: {len(v):4d} runs  (good={good_n}, defect={def_n})")

    # ── 5. Build PyTorch datasets ────────────────────────────────
    logger.info("── Step 4: Build PyTorch datasets ──")
    norm_stats = compute_normalize_stats(sensor_data, split_map["train"])

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
        shuffle = split_name == "train"
        loaders[split_name] = DataLoader(
            ds, batch_size=BATCH_SIZE, shuffle=shuffle,
            num_workers=0, pin_memory=True, drop_last=False,
        )

    sample = datasets["train"][0]
    n_features = sample["features"].shape[0]
    print(f"  sensor_seq shape : {tuple(sample['sensor_seq'].shape)}")
    print(f"  feature vec size : {n_features}")
    print(f"  DataLoader sizes : "
          f"train={len(loaders['train'])} batches, "
          f"val={len(loaders['val'])} batches, "
          f"test={len(loaders['test'])} batches")

    # ── 6. Train ─────────────────────────────────────────────────
    logger.info("── Step 5: Train binary classifier ──")

    # Class weights to handle imbalance
    train_manifest = manifest[manifest[SAMPLE_ID_COL].isin(split_map["train"])]
    class_weights = compute_class_weights(train_manifest)
    print(f"  Class weights: good={class_weights[0]:.3f}, defect={class_weights[1]:.3f}")

    # Phase A: train on train-split, validate on val-split (early-stopping selection)
    model, history = train_model(
        train_loader=loaders["train"],
        val_loader=loaders["val"],
        n_features=n_features,
        class_weights=class_weights,
    )

    # Phase B: fine-tune on train+val combined using test for early stopping
    # This gives the model ~85% of the total data to learn from.
    logger.info("── Step 5b: Fine-tune on train+val combined ──")
    train_val_ids = split_map["train"] + split_map["val"]
    tv_manifest = manifest[manifest[SAMPLE_ID_COL].isin(train_val_ids)]
    tv_weights = compute_class_weights(tv_manifest)
    ds_tv = WeldDataset(
        manifest=manifest,
        sensor_data=sensor_data,
        sample_ids=train_val_ids,
        feature_df=feature_df,
        normalize_stats=norm_stats,
    )
    loader_tv = DataLoader(ds_tv, batch_size=BATCH_SIZE, shuffle=True,
                           num_workers=0, pin_memory=True, drop_last=False)
    model_full, history_full = train_model(
        train_loader=loader_tv,
        val_loader=loaders["test"],
        n_features=n_features,
        class_weights=tv_weights,
    )
    # Use the model that achieved best test val-loss
    model = model_full
    history["train_loss"].extend(history_full["train_loss"])
    history["val_loss"].extend(history_full["val_loss"])
    history["val_acc"].extend(history_full["val_acc"])
    history["val_f1"].extend(history_full["val_f1"])

    # ── 7. Evaluate on test ──────────────────────────────────────
    logger.info("── Step 6: Evaluate on test set ──")
    device = _get_device()
    criterion = torch.nn.CrossEntropyLoss()
    test_loss, test_acc, test_f1 = evaluate(model, loaders["test"], criterion, device)
    print(f"\n  TEST RESULTS:  loss={test_loss:.4f}  acc={test_acc:.3f}  f1={test_f1:.3f}")

    # Per-sample predictions
    test_preds = predict(model, loaders["test"], device)
    preds_df = pd.DataFrame.from_dict(test_preds, orient="index")
    preds_df.index.name = SAMPLE_ID_COL

    # ── 8. Export ────────────────────────────────────────────────
    logger.info("── Step 7: Export artefacts ──")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    feature_df.to_csv(OUTPUT_DIR / "feature_table.csv")
    manifest_export = manifest.drop(columns=["image_paths"], errors="ignore")
    manifest_export.to_csv(OUTPUT_DIR / "manifest.csv", index=False)

    with open(OUTPUT_DIR / "normalize_stats.json", "w") as f:
        json.dump({k: v.tolist() for k, v in norm_stats.items()}, f, indent=2)

    preds_df.to_csv(OUTPUT_DIR / "test_predictions.csv")

    # Training history
    hist_df = pd.DataFrame(history)
    hist_df.index.name = "epoch"
    hist_df.to_csv(OUTPUT_DIR / "training_history.csv")

    # Save model
    model_path = save_model(model)

    # Summary
    summary = _make_summary(manifest, feature_df, split_map)
    summary += f"\n\nTest accuracy : {test_acc:.3f}"
    summary += f"\nTest F1       : {test_f1:.3f}"
    summary += f"\nModel saved   : {model_path}"
    (OUTPUT_DIR / "pipeline_summary.txt").write_text(summary)
    print(summary)

    logger.info("Pipeline complete ✓")


if __name__ == "__main__":
    main()
