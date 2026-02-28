"""
Feature engineering for sensor time-series and images.

Two kinds of feature vectors are produced per run:
  1. **Sensor features** – statistical aggregates computed over the full
     run and over sliding windows.
  2. **Image features** – basic pixel statistics (brightness, contrast,
     edge density) for each image, then aggregated per run.

All features are returned as flat numpy arrays so they can feed directly
into a PyTorch Dataset or an sklearn model.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.config import (
    SENSOR_COLUMNS,
    WINDOW_SIZE,
    WINDOW_STRIDE,
    IMAGE_SIZE,
    FIXED_SEQ_LEN,
)

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Sensor features
# ─────────────────────────────────────────────

_AGG_FUNCS = ["mean", "std", "min", "max", "median"]


def _global_sensor_stats(df: pd.DataFrame) -> Dict[str, float]:
    """Compute global aggregates over each sensor column."""
    feats: Dict[str, float] = {}
    for col in SENSOR_COLUMNS:
        series = df[col].dropna()
        for fn in _AGG_FUNCS:
            feats[f"{col}__{fn}"] = float(getattr(series, fn)()) if len(series) else 0.0
        # Extra: range and IQR
        feats[f"{col}__range"] = float(series.max() - series.min()) if len(series) else 0.0
        q75, q25 = series.quantile(0.75), series.quantile(0.25)
        feats[f"{col}__iqr"] = float(q75 - q25) if len(series) else 0.0
    return feats


def _windowed_sensor_stats(
    df: pd.DataFrame,
    window: int = WINDOW_SIZE,
    stride: int = WINDOW_STRIDE,
) -> Dict[str, float]:
    """Sliding-window std/mean ratio (coefficient of variation) – captures
    local instability that might indicate defects."""
    feats: Dict[str, float] = {}
    for col in SENSOR_COLUMNS:
        vals: np.ndarray = df[col].dropna().to_numpy(dtype=np.float64)
        if len(vals) < window:
            win_means = [np.mean(vals)] if len(vals) else [0.0]
            win_stds = [np.std(vals)] if len(vals) else [0.0]
        else:
            starts = range(0, len(vals) - window + 1, stride)
            win_means = [np.mean(vals[s : s + window]) for s in starts]
            win_stds = [np.std(vals[s : s + window]) for s in starts]

        feats[f"{col}__win_mean_of_std"] = float(np.mean(win_stds))
        feats[f"{col}__win_std_of_mean"] = float(np.std(win_means))
        feats[f"{col}__win_max_std"] = float(np.max(win_stds))
    return feats


def _rate_of_change_features(df: pd.DataFrame) -> Dict[str, float]:
    """First-difference stats to capture dynamics."""
    feats: Dict[str, float] = {}
    for col in SENSOR_COLUMNS:
        diff = df[col].diff().dropna()
        if len(diff) == 0:
            for fn in ("mean", "std", "max", "min"):
                feats[f"{col}__diff_{fn}"] = 0.0
            continue
        feats[f"{col}__diff_mean"] = float(diff.mean())
        feats[f"{col}__diff_std"] = float(diff.std())
        feats[f"{col}__diff_max"] = float(diff.max())
        feats[f"{col}__diff_min"] = float(diff.min())
    return feats


def _weld_phase_features(df: pd.DataFrame) -> Dict[str, float]:
    """Extract features based on weld phases: idle / ramp-up /
    steady-state / ramp-down / cool-down detected via Primary Weld Current."""
    current: np.ndarray = df["Primary Weld Current"].to_numpy(dtype=np.float64)
    threshold = 10.0  # amps – above this the arc is on

    arcing: np.ndarray = current > threshold
    n_total = len(current)
    n_arcing = int(arcing.sum())

    feats: Dict[str, float] = {
        "phase__arc_fraction": n_arcing / n_total if n_total else 0.0,
        "phase__n_arcing_rows": n_arcing,
        "phase__n_idle_rows": n_total - n_arcing,
    }

    # First and last arcing index
    arc_indices = np.where(arcing)[0]
    if len(arc_indices) > 0:
        feats["phase__arc_start_idx"] = int(arc_indices[0]) / n_total
        feats["phase__arc_end_idx"] = int(arc_indices[-1]) / n_total
        feats["phase__arc_duration_frac"] = (arc_indices[-1] - arc_indices[0]) / n_total
    else:
        feats["phase__arc_start_idx"] = 0.0
        feats["phase__arc_end_idx"] = 0.0
        feats["phase__arc_duration_frac"] = 0.0

    return feats


def extract_sensor_features(df: pd.DataFrame) -> Dict[str, float]:
    """Full sensor feature vector for a single run."""
    feats: Dict[str, float] = {}
    feats.update(_global_sensor_stats(df))
    feats.update(_windowed_sensor_stats(df))
    feats.update(_rate_of_change_features(df))
    feats.update(_weld_phase_features(df))
    feats["meta__n_rows"] = len(df)
    return feats


# ─────────────────────────────────────────────
# Image features (basic statistics)
# ─────────────────────────────────────────────

def _load_image_gray(path: Path, size: Tuple[int, int] = IMAGE_SIZE) -> Optional[np.ndarray]:
    """Load image as grayscale, resize.  Returns HxW uint8 array or None."""
    try:
        from PIL import Image
        img = Image.open(path).convert("L").resize(size)
        return np.array(img, dtype=np.uint8)
    except Exception as exc:
        logger.warning("Failed to load image %s: %s", path, exc)
        return None


def _single_image_stats(img: np.ndarray) -> Dict[str, float]:
    """Compute basic statistics for one grayscale image."""
    img_f = img.astype(np.float32) / 255.0
    feats: Dict[str, float] = {
        "img__brightness_mean": float(img_f.mean()),
        "img__brightness_std": float(img_f.std()),
        "img__brightness_min": float(img_f.min()),
        "img__brightness_max": float(img_f.max()),
    }

    # Histogram entropy
    hist, _ = np.histogram(img_f.ravel(), bins=64, range=(0, 1))
    hist = hist / hist.sum()
    hist = hist[hist > 0]
    feats["img__hist_entropy"] = float(-np.sum(hist * np.log2(hist)))

    # Simple edge density via Sobel-like horizontal gradient
    grad_x = np.abs(np.diff(img_f, axis=1)).mean()
    grad_y = np.abs(np.diff(img_f, axis=0)).mean()
    feats["img__edge_density"] = float(grad_x + grad_y)

    return feats


def extract_image_features(image_paths: List[Path]) -> Dict[str, float]:
    """Aggregate image features across all images of a run."""
    if not image_paths:
        return {
            "img__n_images": 0,
            "img__brightness_mean": 0.0,
            "img__brightness_std": 0.0,
            "img__brightness_min": 0.0,
            "img__brightness_max": 0.0,
            "img__hist_entropy": 0.0,
            "img__edge_density": 0.0,
        }

    all_stats: List[Dict[str, float]] = []
    for p in image_paths:
        img = _load_image_gray(p)
        if img is not None:
            all_stats.append(_single_image_stats(img))

    if not all_stats:
        return {"img__n_images": 0}

    # Average over all images in the run
    agg: Dict[str, float] = {"img__n_images": len(all_stats)}
    for key in all_stats[0]:
        values = [s[key] for s in all_stats]
        agg[f"{key}__run_mean"] = float(np.mean(values))
        agg[f"{key}__run_std"] = float(np.std(values))

    return agg


# ─────────────────────────────────────────────
# Sensor time-series → fixed-length tensor
# ─────────────────────────────────────────────

def sensor_to_fixed_tensor(
    df: pd.DataFrame,
    seq_len: int = FIXED_SEQ_LEN,
) -> np.ndarray:
    """Convert sensor DataFrame to a (seq_len, n_channels) float32 array.
    Pads with zeros or truncates to ensure uniform length."""
    vals = df[SENSOR_COLUMNS].values.astype(np.float32)
    n = vals.shape[0]
    n_ch = vals.shape[1]
    out = np.zeros((seq_len, n_ch), dtype=np.float32)
    if n >= seq_len:
        out[:] = vals[:seq_len]
    else:
        out[:n] = vals
    return out


# ─────────────────────────────────────────────
# Build full feature table
# ─────────────────────────────────────────────

def build_feature_table(
    manifest: pd.DataFrame,
    sensor_data: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Compute features for every run and return a flat feature DataFrame
    indexed by sample_id."""
    rows = []
    for _, row in manifest.iterrows():
        sid = row["sample_id"]
        sdf = sensor_data[sid]
        feats = extract_sensor_features(sdf)

        img_paths = row.get("image_paths", [])
        if isinstance(img_paths, list):
            feats.update(extract_image_features(img_paths))

        feats["sample_id"] = sid
        rows.append(feats)

    ft = pd.DataFrame(rows).set_index("sample_id")
    logger.info("Feature table: %d samples × %d features", *ft.shape)
    return ft
