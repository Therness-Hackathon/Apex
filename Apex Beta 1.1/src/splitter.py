"""
Group-aware train / validation / test splitter.

Splitting by **category folder** (weld session) so all runs from the
same physical session stay together – prevents data leakage from
near-duplicate temporal segments.

Stratification ensures the class ratio is roughly preserved in each
split.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from src.config import (
    SAMPLE_ID_COL,
    LABEL_COL,
    CATEGORY_COL,
    SPLIT_DIR,
    TRAIN_RATIO,
    VAL_RATIO,
    TEST_RATIO,
    RANDOM_SEED,
)

logger = logging.getLogger(__name__)


def group_split(
    manifest: pd.DataFrame,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    test_ratio: float = TEST_RATIO,
    seed: int = RANDOM_SEED,
) -> Dict[str, List[str]]:
    """
    Stratified group split by category folder.

    Groups (category folders) are split while trying to maintain
    the overall class ratio in each subset.  Each category's runs
    go entirely into one of train / val / test.

    Returns
    -------
    dict with keys 'train', 'val', 'test', each mapping to a list of sample_ids.
    """
    rng = np.random.RandomState(seed)
    manifest = manifest.copy()

    # Use category column as group key; fall back to sample_id
    group_col = CATEGORY_COL if CATEGORY_COL in manifest.columns else SAMPLE_ID_COL

    # Build a group → (majority_label, n_runs) mapping
    group_info = (
        manifest.groupby(group_col)
        .agg(majority_label=(LABEL_COL, lambda x: int(x.mode().iloc[0])),
             n_runs=(SAMPLE_ID_COL, "count"))
        .reset_index()
    )

    # Stratify: split good-groups and defect-groups separately
    split_map: Dict[str, List[str]] = {"train": [], "val": [], "test": []}

    for lbl in sorted(group_info["majority_label"].unique()):
        grps = group_info[group_info["majority_label"] == lbl][group_col].tolist()
        rng.shuffle(grps)
        n = len(grps)
        n_train = max(1, int(round(n * train_ratio)))
        n_val = max(1, int(round(n * val_ratio)))

        train_grps = set(grps[:n_train])
        val_grps = set(grps[n_train : n_train + n_val])
        test_grps = set(grps[n_train + n_val :])

        for _, row in manifest[manifest[group_col].isin(grps)].iterrows():
            sid = row[SAMPLE_ID_COL]
            g = row[group_col]
            if g in train_grps:
                split_map["train"].append(sid)
            elif g in val_grps:
                split_map["val"].append(sid)
            else:
                split_map["test"].append(sid)

    for k, v in split_map.items():
        logger.info("Split %-5s : %d runs", k, len(v))

    return split_map


def save_split(split_map: Dict[str, List[str]], out_dir: Path = SPLIT_DIR) -> Path:
    """Persist split to JSON for reproducibility."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "split.json"
    with open(path, "w") as f:
        json.dump(split_map, f, indent=2)
    logger.info("Split saved to %s", path)
    return path


def load_split(path: Path = SPLIT_DIR / "split.json") -> Dict[str, List[str]]:
    """Load a previously saved split."""
    with open(path) as f:
        return json.load(f)
