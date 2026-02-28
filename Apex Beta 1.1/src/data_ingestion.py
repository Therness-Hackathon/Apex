"""
Data ingestion & validation layer – v2
=======================================
Handles the real dataset layout:

    good_weld/
        <category_folder>/      (e.g. good_weld_08_11_22_butt_joint)
            <run_folder>/       (e.g. 08-11-22-0001-00)
                <run_id>.csv
                <run_id>.avi
                <run_id>.flac
                images/
                    *.jpg

    defect-weld/
        <category_folder>/      (e.g. burnthrough_weld_1_09_09_2022_butt_joint)
            <run_folder>/
                …

Labels are derived automatically from the top-level folder:
  good_weld  → label 0 (good)
  defect-weld → label 1 (defect)

Defect sub-types are parsed from the category folder name prefix.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

from src.config import (
    GOOD_WELD_DIR,
    DEFECT_WELD_DIR,
    LABELS_CSV,
    SENSOR_COLUMNS,
    TIMESTAMP_COLS,
    PART_NO_COL,
    REMARKS_COL,
    LABEL_COL,
    LABEL_INV,
    SAMPLE_ID_COL,
    CATEGORY_COL,
    DEFECT_TYPE_COL,
    DEFECT_TYPES,
)

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Discovery
# ──────────────────────────────────────────────

def _discover_runs_in(root: Path, label: int) -> List[Dict]:
    """Walk root/<category>/<run>/ and collect metadata dicts."""
    results: List[Dict] = []
    if not root.exists():
        logger.warning("Data directory does not exist: %s", root)
        return results

    for cat_dir in sorted(root.iterdir()):
        if not cat_dir.is_dir():
            continue
        cat_name = cat_dir.name
        # Skip non-data folders like wave_files
        if cat_name in {"wave_files", "__pycache__", ".git"}:
            continue

        # Derive defect sub-type from category name prefix
        defect_type = "good"
        if label == 1:
            for prefix, dtype in DEFECT_TYPES.items():
                if cat_name.lower().startswith(prefix):
                    defect_type = dtype
                    break
            else:
                defect_type = "unknown_defect"

        # Each child of the category folder should be a run
        for run_dir in sorted(cat_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            csv_path = run_dir / f"{run_dir.name}.csv"
            if not csv_path.exists():
                continue
            results.append({
                "run_dir": run_dir,
                SAMPLE_ID_COL: run_dir.name,
                CATEGORY_COL: cat_name,
                LABEL_COL: label,
                DEFECT_TYPE_COL: defect_type,
            })

    logger.info("Discovered %d runs in %s (label=%d)", len(results), root.name, label)
    return results


def discover_all_runs(
    good_dir: Path = GOOD_WELD_DIR,
    defect_dir: Path = DEFECT_WELD_DIR,
) -> List[Dict]:
    """Discover runs from both good_weld/ and defect-weld/."""
    runs = _discover_runs_in(good_dir, label=0)
    runs += _discover_runs_in(defect_dir, label=1)
    logger.info("Total runs discovered: %d", len(runs))
    return runs


# ──────────────────────────────────────────────
# CSV parsing
# ──────────────────────────────────────────────

def _parse_sensor_csv(csv_path: Path) -> pd.DataFrame:
    """Load a single run's sensor CSV, normalise column names, add
    datetime and sample_id columns."""
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    if "Date" in df.columns and "Time" in df.columns:
        df["datetime"] = pd.to_datetime(
            df["Date"].astype(str) + " " + df["Time"].astype(str),
            format="mixed",
            dayfirst=False,
        )
    else:
        df["datetime"] = pd.NaT

    df[SAMPLE_ID_COL] = csv_path.stem
    return df


def _catalogue_images(images_dir: Path) -> List[Path]:
    """Return sorted paths of images inside a run's images/ folder."""
    if not images_dir.exists():
        return []
    return sorted(
        p for p in images_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    )


# ──────────────────────────────────────────────
# Validation
# ──────────────────────────────────────────────

def _validate_run(
    sample_id: str,
    sensor_df: pd.DataFrame,
    image_paths: List[Path],
) -> Dict:
    """Quality-check dict for a single run."""
    issues: List[str] = []

    n_rows = len(sensor_df)
    if n_rows == 0:
        issues.append("empty_csv")

    missing = sensor_df[SENSOR_COLUMNS].isnull().sum()
    cols_with_nan = list(missing[missing > 0].index)
    if cols_with_nan:
        issues.append(f"missing_values:{cols_with_nan}")

    std = sensor_df[SENSOR_COLUMNS].std()
    const_cols = list(std[std == 0.0].index)

    if "datetime" in sensor_df.columns and sensor_df["datetime"].notna().any():
        duration_s = (
            sensor_df["datetime"].max() - sensor_df["datetime"].min()
        ).total_seconds()
    else:
        duration_s = np.nan

    n_images = len(image_paths)
    if n_images == 0:
        issues.append("no_images")

    return {
        "n_sensor_rows": n_rows,
        "duration_s": round(duration_s, 2) if not np.isnan(duration_s) else None,
        "n_images": n_images,
        "const_sensor_cols": const_cols,
        "issues": issues,
    }


# ──────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────

def ingest(
    good_dir: Path = GOOD_WELD_DIR,
    defect_dir: Path = DEFECT_WELD_DIR,
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Main ingestion entry-point for the real dataset.

    Returns
    -------
    manifest : pd.DataFrame
        One row per run with sample_id, label, defect_type, category,
        quality-check columns, and image_paths.
    sensor_data : dict[str, pd.DataFrame]
        Mapping sample_id → raw sensor DataFrame.
    """
    run_records = discover_all_runs(good_dir, defect_dir)
    if not run_records:
        raise FileNotFoundError(
            f"No valid run folders discovered in {good_dir} or {defect_dir}"
        )

    sensor_data: Dict[str, pd.DataFrame] = {}
    manifest_rows: List[Dict] = []
    dupes: set = set()

    for rec in run_records:
        run_dir: Path = rec["run_dir"]
        sid: str = rec[SAMPLE_ID_COL]

        # Handle duplicate sample_ids across categories
        if sid in dupes:
            # Prefix with category to disambiguate
            new_sid = f"{rec[CATEGORY_COL]}__{sid}"
            logger.debug("Duplicate sample_id %s → renamed to %s", sid, new_sid)
            sid = new_sid
            rec[SAMPLE_ID_COL] = sid
        dupes.add(sid)

        csv_path = run_dir / f"{run_dir.name}.csv"
        images_dir = run_dir / "images"

        sensor_df = _parse_sensor_csv(csv_path)
        sensor_df[SAMPLE_ID_COL] = sid
        image_paths = _catalogue_images(images_dir)
        sensor_data[sid] = sensor_df

        qc = _validate_run(sid, sensor_df, image_paths)
        row = {
            SAMPLE_ID_COL: sid,
            LABEL_COL: rec[LABEL_COL],
            DEFECT_TYPE_COL: rec[DEFECT_TYPE_COL],
            CATEGORY_COL: rec[CATEGORY_COL],
            "image_paths": image_paths,
        }
        row.update(qc)
        manifest_rows.append(row)

    manifest = pd.DataFrame(manifest_rows)

    n_good = (manifest[LABEL_COL] == 0).sum()
    n_defect = (manifest[LABEL_COL] == 1).sum()
    logger.info(
        "Ingestion complete – %d runs (%d good, %d defect), %d with issues.",
        len(manifest), n_good, n_defect,
        manifest["issues"].apply(len).gt(0).sum(),
    )
    return manifest, sensor_data


def save_labels_csv(manifest: pd.DataFrame, out_path: Path = LABELS_CSV) -> Path:
    """Export auto-generated labels to labels.csv."""
    cols = [SAMPLE_ID_COL, LABEL_COL, DEFECT_TYPE_COL, CATEGORY_COL]
    manifest[cols].to_csv(out_path, index=False)
    logger.info("Saved %d labels to %s", len(manifest), out_path)
    return out_path
