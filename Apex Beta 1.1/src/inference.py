#!/usr/bin/env python3
"""
Phase 2 – Inference script (one-command run).

Usage
-----
    python src/inference.py                          # defaults: test split
    python src/inference.py --split val
    python src/inference.py --csv path/to/run.csv    # single-file mode

Outputs
-------
    outputs/predictions_binary.csv
        Columns: sample_id, p_defect, pred_defect, confidence
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

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    MODEL_DIR, OUTPUT_DIR, SPLIT_DIR, SENSOR_COLUMNS, FIXED_SEQ_LEN,
    BATCH_SIZE, LABEL_MAP, SAMPLE_ID_COL, LABEL_COL,
)
from src.trainer import WeldClassifier, _get_device
from src.calibration import TemperatureScaler
from src.data_ingestion import ingest
from src.feature_engineering import (
    build_feature_table,
    extract_sensor_features,
    extract_image_features,
    sensor_to_fixed_tensor,
)
from src.splitter import load_split
from src.dataset import WeldDataset, compute_normalize_stats

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s")
logger = logging.getLogger("inference")

# ── Default artefact paths ────────────────────────────────────
MODEL_PATH      = MODEL_DIR / "weld_classifier.pt"
TEMP_PATH       = MODEL_DIR / "temperature.json"
THRESHOLD_PATH  = MODEL_DIR / "threshold.json"
NORM_STATS_PATH = OUTPUT_DIR / "normalize_stats.json"
SPLIT_PATH      = SPLIT_DIR / "split.json"
PREDICTIONS_OUT = OUTPUT_DIR / "predictions_binary.csv"


def _load_model(device: torch.device, n_features: int = 104) -> WeldClassifier:
    """Load trained model checkpoint."""
    n_channels = len(SENSOR_COLUMNS)
    model = WeldClassifier(n_channels, n_features).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    logger.info("Model loaded from %s", MODEL_PATH)
    return model


def _load_temperature(model: WeldClassifier, device: torch.device) -> TemperatureScaler:
    """Load temperature scaler if available, else return identity (T=1)."""
    scaler = TemperatureScaler(model).to(device)
    if TEMP_PATH.exists():
        with open(TEMP_PATH) as f:
            T = json.load(f)["temperature"]
        scaler.temperature.data.fill_(T)
        logger.info("Temperature loaded: T=%.4f", T)
    else:
        logger.warning("No temperature file found at %s – using T=1.0 (uncalibrated)", TEMP_PATH)
    return scaler


def _load_threshold() -> float:
    """Load the fixed decision threshold chosen on validation."""
    if THRESHOLD_PATH.exists():
        with open(THRESHOLD_PATH) as f:
            t = json.load(f)["threshold"]
        logger.info("Decision threshold loaded: %.4f", t)
        return float(t)
    logger.warning("No threshold file – defaulting to 0.50")
    return 0.50


def _load_norm_stats():
    """Load normalisation stats."""
    with open(NORM_STATS_PATH) as f:
        raw = json.load(f)
    return {k: np.array(v, dtype=np.float32) for k, v in raw.items()}


# ──────────────────────────────────────────────────────────────
# Full-split inference
# ──────────────────────────────────────────────────────────────

def run_split_inference(split_name: str = "test") -> pd.DataFrame:
    """Run inference on a full data split. Returns predictions DataFrame."""
    device = _get_device()

    # 1. Ingest data
    from src.config import GOOD_WELD_DIR, DEFECT_WELD_DIR
    manifest, sensor_data = ingest(GOOD_WELD_DIR, DEFECT_WELD_DIR)

    # 2. Features
    feature_df = build_feature_table(manifest, sensor_data)
    n_features = feature_df.shape[1]

    # 3. Split
    split_map = load_split(SPLIT_PATH)
    sample_ids = split_map[split_name]
    logger.info("Split '%s': %d samples", split_name, len(sample_ids))

    # 4. Normalization stats
    norm_stats = _load_norm_stats()

    # 5. Dataset + Loader
    ds = WeldDataset(
        manifest=manifest,
        sensor_data=sensor_data,
        sample_ids=sample_ids,
        feature_df=feature_df,
        normalize_stats=norm_stats,
    )
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    # 6. Model + calibration
    model = _load_model(device, n_features=n_features)
    scaler = _load_temperature(model, device)
    threshold = _load_threshold()

    # 7. Predict
    scaler.eval()
    rows = []
    with torch.no_grad():
        for batch in loader:
            seq = batch["sensor_seq"].to(device)
            feat = batch["features"].to(device)
            sids = batch["sample_id"]
            labels = batch["label"]

            probs = scaler.calibrated_probs(seq, feat)

            for i, sid in enumerate(sids):
                p_defect = float(probs[i, 1].item())
                pred = 1 if p_defect >= threshold else 0
                confidence = p_defect if pred == 1 else (1.0 - p_defect)
                lbl = int(labels[i].item()) if labels[i].item() >= 0 else None

                rows.append({
                    "sample_id": sid,
                    "p_defect": round(p_defect, 6),
                    "pred_defect": pred,
                    "confidence": round(confidence, 6),
                    "label": lbl,
                })

    df = pd.DataFrame(rows)
    return df


# ──────────────────────────────────────────────────────────────
# Single-CSV inference
# ──────────────────────────────────────────────────────────────

def run_single_csv(csv_path: Path) -> dict:
    """Run inference on a single CSV file (no label needed)."""
    device = _get_device()

    # 1. Read CSV
    raw_df = pd.read_csv(csv_path)
    raw_df.columns = raw_df.columns.str.strip()

    # 2. Sensor sequence
    norm_stats = _load_norm_stats()
    seq = sensor_to_fixed_tensor(raw_df, FIXED_SEQ_LEN)
    mu, sd = norm_stats["mean"], norm_stats["std"].copy()
    sd[sd == 0] = 1.0
    seq = (seq - mu) / sd
    sensor_tensor = torch.from_numpy(seq).unsqueeze(0).to(device)

    # 3. Features
    feats = extract_sensor_features(raw_df)
    images_dir = csv_path.parent / "images"
    image_paths = (
        sorted(images_dir.glob("*.jpg")) + sorted(images_dir.glob("*.png"))
        if images_dir.exists() else []
    )
    feats.update(extract_image_features(image_paths))
    feat_vals = np.nan_to_num(
        np.array(list(feats.values()), dtype=np.float32), nan=0.0
    )
    feat_tensor = torch.from_numpy(feat_vals).unsqueeze(0).to(device)

    # 4. Model + calibration
    n_features = feat_tensor.shape[1]
    model = _load_model(device, n_features=n_features)
    scaler = _load_temperature(model, device)
    threshold = _load_threshold()

    # 5. Predict
    scaler.eval()
    with torch.no_grad():
        probs = scaler.calibrated_probs(sensor_tensor, feat_tensor)

    p_defect = float(probs[0, 1].item())
    pred = 1 if p_defect >= threshold else 0
    confidence = p_defect if pred == 1 else (1.0 - p_defect)

    return {
        "csv": str(csv_path),
        "p_defect": round(p_defect, 6),
        "pred_defect": pred,
        "pred_label": LABEL_MAP[pred].upper(),
        "confidence": round(confidence, 6),
        "threshold": threshold,
    }


# ──────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Phase 2 – Weld defect inference")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"],
                        help="Which data split to run inference on (default: test)")
    parser.add_argument("--csv", type=str, default=None,
                        help="Path to a single CSV file for inference")
    parser.add_argument("--out", type=str, default=None,
                        help="Output CSV path (default: outputs/predictions_binary.csv)")
    args = parser.parse_args()

    if args.csv:
        # Single-file mode
        result = run_single_csv(Path(args.csv))
        print(f"\n{'=' * 50}")
        print(f"  CSV        : {result['csv']}")
        print(f"  P(defect)  : {result['p_defect']:.4f}")
        print(f"  Prediction : {result['pred_label']}")
        print(f"  Confidence : {result['confidence']:.2%}")
        print(f"  Threshold  : {result['threshold']:.4f}")
        print(f"{'=' * 50}")
    else:
        # Full-split mode
        df = run_split_inference(args.split)

        out_path = Path(args.out) if args.out else PREDICTIONS_OUT
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)

        n_defect = (df["pred_defect"] == 1).sum()
        print(f"\nPredictions written to {out_path}")
        print(f"  Samples: {len(df)}")
        print(f"  Predicted defect: {n_defect}  ({n_defect / len(df):.1%})")
        print(f"  Predicted good:   {len(df) - n_defect}  ({(len(df) - n_defect) / len(df):.1%})")

        # If labels available, show quick metrics
        labelled = df[df["label"].notna()]
        if len(labelled) > 0:
            from src.evaluation import compute_binary_metrics
            threshold = _load_threshold()
            metrics = compute_binary_metrics(
                np.asarray(labelled["label"].values, dtype=int),
                np.asarray(labelled["p_defect"].values),
                threshold=threshold,
            )
            print(f"\n  Accuracy   : {metrics['accuracy']:.4f}")
            print(f"  F1         : {metrics['f1']:.4f}")
            print(f"  ROC AUC    : {metrics['roc_auc']:.4f}")


if __name__ == "__main__":
    main()
