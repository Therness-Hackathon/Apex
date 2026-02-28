"""
Multi-class defect-type classifier.

Trains a gradient-boosted tree on the pre-computed feature table to predict
one of six label codes.  The binary weld classifier handles good/defect; this
module only predicts WHICH defect type when the binary model fires.

Label-code mapping (matches competition submission schema):
    good                 → "00"
    excessive_penetration → "01"
    burnthrough          → "02"
    overlap              → "06"
    lack_of_fusion       → "07"
    excessive_convexity  → "08"
    crater_cracks        → "11"
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel

from src.config import OUTPUT_DIR, MODEL_DIR, LABELS_CSV

logger = logging.getLogger(__name__)

# ── Label-code mapping ────────────────────────────────────────────────────────
DEFECT_TYPE_TO_CODE: Dict[str, str] = {
    "good":                  "00",
    "excessive_penetration": "01",
    "burnthrough":           "02",
    "lack_of_fusion":        "07",
    "overlap":               "06",
    "excessive_convexity":   "08",
    "crater_cracks":         "11",
}

CODE_TO_DEFECT_TYPE: Dict[str, str] = {v: k for k, v in DEFECT_TYPE_TO_CODE.items()}

ALLOWED_CODES: List[str] = ["00", "01", "02", "06", "07", "08", "11"]

_MC_MODEL_PATH = MODEL_DIR / "multiclass_classifier.pkl"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _feature_matrix(
    feature_table: pd.DataFrame,
    sample_ids: Optional[List[str]] = None,
) -> Tuple[np.ndarray, List[str]]:
    """Return (X, feature_cols) with NaNs filled."""
    drop_cols = {"sample_id", "label", "defect_type", "category", "code"}
    feat_cols = [c for c in feature_table.columns if c not in drop_cols]
    df = feature_table if sample_ids is None else feature_table[
        feature_table["sample_id"].isin(sample_ids)
    ]
    X = df[feat_cols].fillna(0.0).to_numpy(dtype=np.float32)
    return X, feat_cols


# ── Training ──────────────────────────────────────────────────────────────────

def train_multiclass(
    feature_csv: Path = OUTPUT_DIR / "multimodal_features.csv",
    labels_csv: Path = LABELS_CSV,
    split_json: Optional[Path] = None,
    save_path: Path = _MC_MODEL_PATH,
) -> Pipeline:
    """Train a regularised multi-class defect-type classifier.

    Strategy:
    - Train on ALL available samples (train + val + test)
      so the competition test samples see maximum training coverage.
    - Use a regularised GBM (shallow trees, feature subsampling, min_samples_leaf)
      to prevent the 1.00/0.08 overfitting gap seen previously.
    - Report 5-fold stratified CV macro-F1 to give a honest estimate.
    - Save best model.
    """
    # Fall back to sensor-only features if multimodal file doesn't exist yet
    if not feature_csv.exists():
        fallback = OUTPUT_DIR / "feature_table.csv"
        logger.warning(
            "Multimodal features not found at %s — falling back to %s",
            feature_csv, fallback,
        )
        feature_csv = fallback

    feat = pd.read_csv(feature_csv)
    labels = pd.read_csv(labels_csv)[["sample_id", "defect_type"]]
    feat = feat.merge(labels, on="sample_id", how="left")

    # Use all data (no artificial hold-out — test is truly held by competition)
    if split_json is not None and split_json.exists():
        with open(split_json) as fh:
            split = json.load(fh)
        # Use train + val + test from our internal split = all 1551
        all_ids = (
            set(split.get("train", []))
            | set(split.get("val", []))
            | set(split.get("test", []))
        )
        feat = feat[feat["sample_id"].isin(all_ids)]

    # Map defect_type → code
    feat["code"] = feat["defect_type"].map(DEFECT_TYPE_TO_CODE)
    feat = feat.dropna(subset=["code"])

    X, feat_cols = _feature_matrix(feat)
    y = feat["code"].to_numpy()

    classes, counts = np.unique(y, return_counts=True)
    logger.info(
        "Training multi-class classifier on %d samples, classes=%s",
        len(y), list(zip(classes, counts)),
    )

    # ── Define regularised pipeline ───────────────────────────────────────────
    # Shallow trees + row/col subsampling + minimum leaf size → much less overfit
    gbm = GradientBoostingClassifier(
        n_estimators=150,
        max_depth=3,           # was 4
        learning_rate=0.05,    # slower learning
        subsample=0.7,         # row subsampling
        max_features=0.6,      # column subsampling per split
        min_samples_leaf=15,   # minimum samples per leaf
        min_samples_split=30,  # minimum samples to split
        random_state=42,
    )

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", gbm),
    ])

    # ── 5-fold stratified CV to estimate true generalisation ─────────────────
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipe, X, y, cv=cv, scoring="f1_macro", n_jobs=-1)
    logger.info(
        "5-fold CV macro-F1: %.4f ± %.4f  (folds: %s)",
        cv_scores.mean(), cv_scores.std(),
        [f"{s:.3f}" for s in cv_scores],
    )

    # ── Fit on all data ───────────────────────────────────────────────────────
    pipe.fit(X, y)

    # Save
    save_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, save_path)
    logger.info("Saved multi-class model → %s", save_path)

    # CV score is the real estimate; in-sample is expected to look good
    y_pred_train = pipe.predict(X)
    train_macro = f1_score(y, y_pred_train, average="macro")
    logger.info(
        "Train macro-F1 (in-sample): %.4f  |  CV macro-F1 (honest est): %.4f",
        train_macro, cv_scores.mean(),
    )
    logger.info(
        "Train classification report:\n%s",
        classification_report(y, y_pred_train, target_names=sorted(np.unique(y).tolist())),
    )
    return pipe


# ── Inference ─────────────────────────────────────────────────────────────────

def load_multiclass_model(path: Path = _MC_MODEL_PATH) -> Pipeline:
    """Load persisted multi-class model, training it first if not found."""
    if not path.exists():
        logger.warning(
            "Multi-class model not found at %s – training now.", path
        )
        from src.config import SPLIT_DIR
        train_multiclass(split_json=SPLIT_DIR / "split.json")
    return joblib.load(path)


def predict_type_codes(
    sample_ids: List[str],
    feature_table: pd.DataFrame,
    binary_preds: Optional[Dict[str, int]] = None,
    model_path: Path = _MC_MODEL_PATH,
) -> Dict[str, str]:
    """Return {sample_id: pred_label_code} for every requested sample.

    If *binary_preds* is supplied, samples predicted as GOOD (0) are
    unconditionally assigned code "00" instead of running the multi-class
    model.
    """
    pipe = load_multiclass_model(model_path)

    # Filter feature table to requested samples
    ft = feature_table[feature_table["sample_id"].isin(sample_ids)].copy()

    results: Dict[str, str] = {}

    if len(ft) == 0:
        logger.warning("No feature rows found for %d requested sample IDs.", len(sample_ids))
        for sid in sample_ids:
            results[sid] = "00"
        return results

    # Identify which samples need multi-class inference
    if binary_preds is not None:
        good_ids = {sid for sid in sample_ids if binary_preds.get(sid, 0) == 0}
        defect_ids = [sid for sid in sample_ids if sid not in good_ids]
    else:
        good_ids = set()
        defect_ids = list(sample_ids)

    # Good samples → code "00"
    for sid in good_ids:
        results[sid] = "00"

    # Defect samples → run multi-class
    if defect_ids:
        ft_defect = ft[ft["sample_id"].isin(defect_ids)]
        if len(ft_defect) > 0:
            X, _ = _feature_matrix(ft_defect)
            proba = pipe.predict_proba(X)
            classes = pipe.classes_
            good_idx = list(classes).index("00") if "00" in classes else None
            for i, sid in enumerate(ft_defect["sample_id"].tolist()):
                row_proba = proba[i].copy()
                # Suppress the "00" (good) class — the binary model already decided defect
                if good_idx is not None:
                    row_proba[good_idx] = 0.0
                code = classes[int(np.argmax(row_proba))]
                results[sid] = code
        # Any defect IDs missing from feature table default to "02" (burnthrough)
        for sid in defect_ids:
            if sid not in results:
                results[sid] = "02"

    return results
