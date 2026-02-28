#!/usr/bin/env python3
"""
Multimodal feature extraction — audio (.flac) + video (.avi) + existing sensor features.

Scans every run folder in good_weld/ and defect-weld/, extracts audio and video
features, then merges them with the pre-computed sensor feature_table.csv.

The result is saved as outputs/multimodal_features.csv and used by the
multiclass classifier and the WeldDataset feature-MLP branch.

Usage:
    python extract_multimodal.py              # extract ALL samples
    python extract_multimodal.py --limit 50   # test run on first 50 samples
    python extract_multimodal.py --force       # re-extract even if cache exists
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import GOOD_WELD_DIR, DEFECT_WELD_DIR, OUTPUT_DIR
from src.audio_features import extract_audio_features
from src.video_features import extract_video_features

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger("extract_multimodal")

SENSOR_FEATURE_CSV = OUTPUT_DIR / "feature_table.csv"
OUTPUT_CSV         = OUTPUT_DIR / "multimodal_features.csv"


# ─────────────────────────────────────────────────────────────────────────────
# Discovery
# ─────────────────────────────────────────────────────────────────────────────

def _discover_runs(roots: List[Path]) -> List[Dict]:
    """Return list of {sample_id, flac_path, avi_path} for every run found."""
    runs = []
    skip = {"wave_files", "__pycache__", ".git"}
    for root in roots:
        if not root.exists():
            logger.warning("Directory missing: %s", root)
            continue
        for cat_dir in sorted(root.iterdir()):
            if not cat_dir.is_dir() or cat_dir.name in skip:
                continue
            for run_dir in sorted(cat_dir.iterdir()):
                if not run_dir.is_dir():
                    continue
                sid = run_dir.name
                runs.append({
                    "sample_id": sid,
                    "flac_path": run_dir / f"{sid}.flac",
                    "avi_path":  run_dir / f"{sid}.avi",
                })
    logger.info("Discovered %d runs", len(runs))
    return runs


# ─────────────────────────────────────────────────────────────────────────────
# Main extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_all(runs: List[Dict]) -> pd.DataFrame:
    """Extract audio + video features for all runs. Returns DataFrame."""
    rows = []
    n_audio_ok = 0
    n_video_ok = 0

    for run in tqdm(runs, desc="Extracting A/V features"):
        sid       = run["sample_id"]
        aud_feats = extract_audio_features(run["flac_path"])
        vid_feats = extract_video_features(run["avi_path"])

        if aud_feats.get("aud__file_ok", 0.0) > 0:
            n_audio_ok += 1
        if vid_feats.get("vid__file_ok", 0.0) > 0:
            n_video_ok += 1

        row = {"sample_id": sid}
        row.update(aud_feats)
        row.update(vid_feats)
        rows.append(row)

    df = pd.DataFrame(rows)
    logger.info(
        "Audio OK: %d/%d  |  Video OK: %d/%d",
        n_audio_ok, len(runs), n_video_ok, len(runs),
    )
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Merge with sensor features
# ─────────────────────────────────────────────────────────────────────────────

def merge_with_sensor_features(av_df: pd.DataFrame) -> pd.DataFrame:
    """Merge A/V features with existing sensor feature_table.csv.

    Samples present in sensor table but missing from A/V extraction get
    all-zero A/V features (graceful degradation).
    Samples present in A/V extraction but NOT in sensor table are dropped
    (no sensor data = can't train).
    """
    if not SENSOR_FEATURE_CSV.exists():
        logger.error("Sensor feature table not found: %s", SENSOR_FEATURE_CSV)
        sys.exit(1)

    sensor_df = pd.read_csv(SENSOR_FEATURE_CSV)

    # Normalize index column
    if "Unnamed: 0" in sensor_df.columns:
        sensor_df = sensor_df.drop(columns=["Unnamed: 0"])
    if sensor_df.index.name == "sample_id":
        sensor_df = sensor_df.reset_index()

    logger.info("Sensor feature table: %d rows × %d cols", *sensor_df.shape)

    # Left-join: keep all sensor rows, attach A/V where available
    merged = sensor_df.merge(av_df, on="sample_id", how="left")

    # Fill NaN A/V features with 0
    av_cols = [c for c in av_df.columns if c != "sample_id"]
    merged[av_cols] = merged[av_cols].fillna(0.0)

    logger.info(
        "Merged table: %d rows × %d cols  (%d A/V features added)",
        len(merged), merged.shape[1], len(av_cols),
    )
    return merged


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only first N samples (for testing).")
    parser.add_argument("--force", action="store_true",
                        help="Re-extract even if output CSV already exists.")
    args = parser.parse_args()

    if OUTPUT_CSV.exists() and not args.force:
        logger.info(
            "Output already exists at %s  (use --force to re-extract)", OUTPUT_CSV
        )
        return

    # Discover runs
    runs = _discover_runs([GOOD_WELD_DIR, DEFECT_WELD_DIR])

    # Deduplicate by sample_id (keep first occurrence)
    seen = set()
    unique_runs = []
    for r in runs:
        if r["sample_id"] not in seen:
            seen.add(r["sample_id"])
            unique_runs.append(r)
    logger.info("Unique runs after dedup: %d", len(unique_runs))

    if args.limit:
        unique_runs = unique_runs[: args.limit]
        logger.info("Limiting to %d runs (--limit)", args.limit)

    # Extract
    av_df = extract_all(unique_runs)

    # Merge with sensor features
    merged = merge_with_sensor_features(av_df)

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    merged.to_csv(OUTPUT_CSV, index=False)
    logger.info("Saved multimodal features → %s  (%d rows × %d cols)",
                OUTPUT_CSV, len(merged), merged.shape[1])

    # Print summary
    aud_cols = [c for c in merged.columns if c.startswith("aud__")]
    vid_cols = [c for c in merged.columns if c.startswith("vid__")]
    sen_cols = [c for c in merged.columns if not c.startswith(("aud__", "vid__", "sample_id"))]

    print(f"\n  Sensor features : {len(sen_cols)}")
    print(f"  Audio  features : {len(aud_cols)}")
    print(f"  Video  features : {len(vid_cols)}")
    print(f"  Total  features : {len(sen_cols) + len(aud_cols) + len(vid_cols)}")
    print(f"\n  Saved -> {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
