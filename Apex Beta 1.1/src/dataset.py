"""
PyTorch Dataset for Apex weld quality data.

Each sample yields:
  - sensor_seq  : (FIXED_SEQ_LEN, n_channels) float32 tensor
  - features    : (n_features,)              float32 tensor
  - label       : scalar int64 tensor (0=good, 1=defect)  or -1 if unlabelled
  - sample_id   : str
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.config import LABEL_COL, LABEL_INV, SAMPLE_ID_COL, FIXED_SEQ_LEN, SENSOR_COLUMNS
from src.feature_engineering import (
    extract_sensor_features,
    extract_image_features,
    sensor_to_fixed_tensor,
)

logger = logging.getLogger(__name__)


class WeldDataset(Dataset):
    """
    Map-style PyTorch dataset.

    Parameters
    ----------
    manifest : pd.DataFrame
        Must contain at least ``sample_id``, ``label``, ``image_paths``.
    sensor_data : dict
        Mapping sample_id → raw sensor DataFrame.
    sample_ids : list[str]
        Subset of sample_ids to include (from split).
    feature_df : pd.DataFrame or None
        Pre-computed flat features indexed by sample_id.
        If None, features are computed on the fly (slow).
    normalize_stats : dict or None
        {'mean': np.ndarray, 'std': np.ndarray} for sensor z-scoring.
        If None, raw values are used.
    """

    def __init__(
        self,
        manifest: pd.DataFrame,
        sensor_data: Dict[str, pd.DataFrame],
        sample_ids: List[str],
        feature_df: Optional[pd.DataFrame] = None,
        normalize_stats: Optional[Dict[str, np.ndarray]] = None,
    ):
        self.manifest = manifest.set_index(SAMPLE_ID_COL) if SAMPLE_ID_COL in manifest.columns else manifest
        self.sensor_data = sensor_data
        self.sample_ids = sample_ids
        self.feature_df = feature_df
        self.normalize_stats = normalize_stats

    def __len__(self) -> int:
        return len(self.sample_ids)

    def _get_label(self, sid: str) -> int:
        if sid in self.manifest.index and LABEL_COL in self.manifest.columns:
            lbl = self.manifest.loc[sid, LABEL_COL]
            if pd.notna(lbl):
                # Handle both numeric (0/1) and string ('good'/'defect') labels
                if isinstance(lbl, str):
                    return LABEL_INV.get(lbl.strip().lower(), -1)
                try:
                    return int(lbl)  # type: ignore[arg-type]
                except (ValueError, TypeError):
                    return -1
        return -1  # unlabelled sentinel

    def __getitem__(self, idx: int):
        sid = self.sample_ids[idx]
        sdf = self.sensor_data[sid]

        # 1. Fixed-length sensor sequence tensor
        seq = sensor_to_fixed_tensor(sdf, FIXED_SEQ_LEN)  # (T, C)
        if self.normalize_stats is not None:
            mu = self.normalize_stats["mean"]
            sd = self.normalize_stats["std"]
            sd[sd == 0] = 1.0
            seq = (seq - mu) / sd
        sensor_tensor = torch.from_numpy(seq)  # (T, C)

        # 2. Flat feature vector
        if self.feature_df is not None and sid in self.feature_df.index:
            feat_vals = self.feature_df.loc[sid].values.astype(np.float32)
        else:
            feats = extract_sensor_features(sdf)
            img_paths = []
            if sid in self.manifest.index:
                ip = self.manifest.loc[sid].get("image_paths", [])
                if isinstance(ip, list):
                    img_paths = ip
            feats.update(extract_image_features(img_paths))
            feat_vals = np.array(list(feats.values()), dtype=np.float32)

        feat_tensor = torch.from_numpy(np.nan_to_num(np.asarray(feat_vals), nan=0.0))

        # 3. Label
        label = self._get_label(sid)
        label_tensor = torch.tensor(label, dtype=torch.long)

        return {
            "sensor_seq": sensor_tensor,
            "features": feat_tensor,
            "label": label_tensor,
            "sample_id": sid,
        }


def compute_normalize_stats(
    sensor_data: Dict[str, pd.DataFrame],
    sample_ids: List[str],
) -> Dict[str, np.ndarray]:
    """Compute per-channel mean/std across the listed runs (train set only)."""
    all_vals = []
    for sid in sample_ids:
        arr = sensor_data[sid][SENSOR_COLUMNS].values.astype(np.float32)
        all_vals.append(arr)
    stacked = np.concatenate(all_vals, axis=0)
    return {
        "mean": stacked.mean(axis=0),
        "std": stacked.std(axis=0),
    }
