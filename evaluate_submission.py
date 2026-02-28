#!/usr/bin/env python3
"""
Evaluate a competition submission CSV against ground-truth labels.

Computes all required metrics:
  A) Binary (defect vs good)
       - F1 (defect as positive class)  ← primary scoring metric
       - Precision / Recall / Specificity
       - ROC-AUC, PR-AUC
       - Confusion matrix (TP / FP / FN / TN)

  B) Multi-class defect type
       - Macro F1  ← "Type_MacroF1" used in FinalScore
       - Weighted F1
       - Per-class Precision / Recall / F1

  C) Confidence calibration
       - ECE (Expected Calibration Error) on binary task
       - Brier Score

  D) Final composite score
       FinalScore = 0.6 * Binary_F1 + 0.4 * Type_MacroF1

Usage
-----
    python evaluate_submission.py
    python evaluate_submission.py --submission outputs/submission.csv
    python evaluate_submission.py --submission outputs/submission.csv \
                                   --ground-truth outputs/test_labels_90.csv

The script auto-discovers ground truth from labels.csv + split.json when
--ground-truth is not provided, mapping internal weld IDs → sample_XXXX.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    brier_score_loss,
)

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import OUTPUT_DIR, LABELS_CSV, SPLIT_DIR
from src.multiclass import DEFECT_TYPE_TO_CODE, CODE_TO_DEFECT_TYPE, ALLOWED_CODES

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s")
logger = logging.getLogger("evaluate_submission")

SUBMISSION_DEFAULT = OUTPUT_DIR / "submission.csv"


# ── ECE ───────────────────────────────────────────────────────────────────────

def expected_calibration_error(
    y_true: np.ndarray,
    p_pred: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Compute Expected Calibration Error (ECE) over *n_bins* equal-width bins."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (p_pred >= lo) & (p_pred < hi)
        if mask.sum() == 0:
            continue
        acc_bin  = float(y_true[mask].mean())
        conf_bin = float(p_pred[mask].mean())
        ece += mask.sum() / n * abs(acc_bin - conf_bin)
    return float(ece)


# ── Ground-truth helpers ──────────────────────────────────────────────────────

def _build_ground_truth_from_internal(
    submission: pd.DataFrame,
    submission_csv: Optional[Path],
) -> Optional[pd.DataFrame]:
    """Try to reconstruct ground truth from labels.csv.

    Priority:
    1. Use the _manifest CSV saved alongside submission (most reliable).
    2. Fall back to resolving via _weld_id column in submission.
    3. Fall back to resolving first-N rows of test_predictions.csv.
    """
    if submission_csv is None:
        return None

    from src.multiclass import DEFECT_TYPE_TO_CODE
    labels = pd.read_csv(LABELS_CSV)[["sample_id", "defect_type", "label"]]
    labels["code"] = labels["defect_type"].map(DEFECT_TYPE_TO_CODE)

    # --- Priority 0: audio_features_test.csv with weld_id suffix (competition test) ---
    audio_test_path = OUTPUT_DIR / "audio_features_test.csv"
    if audio_test_path.exists():
        aud = pd.read_csv(audio_test_path, usecols=lambda c: c in {"sample_id", "weld_id"})
        if "weld_id" in aud.columns and "sample_id" in aud.columns:
            aud = aud.dropna(subset=["weld_id"]).copy()
            aud["true_label_code"] = aud["weld_id"].astype(str).apply(
                lambda x: x.split("-")[-1].zfill(2)
            )
            aud["true_label_binary"] = (aud["true_label_code"] != "00").astype(int)
            gt_aud = aud[["sample_id", "true_label_binary", "true_label_code"]]
            # Only return if we can match at least the majority of submission rows
            n_match = gt_aud["sample_id"].isin(submission["sample_id"]).sum()
            if n_match >= len(submission) * 0.5:
                logger.info(
                    "Ground truth derived from audio_features_test.csv weld_id suffixes "
                    "(%d rows, %d matched).", len(gt_aud), n_match
                )
                return gt_aud

    # --- Try manifest file next ---
    manifest_path = submission_csv.with_name(submission_csv.stem + "_manifest.csv")
    if manifest_path.exists():
        mf = pd.read_csv(manifest_path, dtype={"pred_label_code": str})
        mf["pred_label_code"] = mf["pred_label_code"].str.zfill(2)
        mf_merged = mf.merge(
            labels.rename(columns={"sample_id": "_weld_id"}),
            on="_weld_id", how="left"
        )
        if not mf_merged["label"].isna().all():
            gt = mf_merged[["sample_id", "label", "code"]].copy()
            gt.columns = ["sample_id", "true_label_binary", "true_label_code"]
            gt["true_label_binary"] = gt["true_label_binary"].fillna(0).astype(int)
            gt["true_label_code"] = gt["true_label_code"].fillna("00")
            logger.info("Ground truth derived from manifest file (%d rows).", len(gt))
            return gt

    # --- Fall back: _weld_id in submission ---
    if "_weld_id" in submission.columns:
        merged = submission.merge(
            labels.rename(columns={"sample_id": "_weld_id"}),
            on="_weld_id", how="left"
        )
        if not merged["code"].isna().all():
            gt = merged[["sample_id", "label", "code"]].copy()
            gt.columns = ["sample_id", "true_label_binary", "true_label_code"]
            gt["true_label_binary"] = gt["true_label_binary"].fillna(0).astype(int)
            gt["true_label_code"] = gt["true_label_code"].fillna("00")
            return gt

    # --- Last resort: order-based match with test_predictions.csv ---
    test_pred = OUTPUT_DIR / "test_predictions.csv"
    if not test_pred.exists():
        return None
    tp = pd.read_csv(test_pred)
    n = len(submission)
    if len(tp) < n:
        return None
    submission = submission.copy()
    submission["_weld_id"] = tp.head(n)["sample_id"].values

    merged = submission.merge(
        labels.rename(columns={"sample_id": "_weld_id"}),
        on="_weld_id", how="left"
    )
    if merged["code"].isna().all():
        return None

    gt = merged[["sample_id", "label", "code"]].copy()
    gt.columns = ["sample_id", "true_label_binary", "true_label_code"]
    gt["true_label_binary"] = gt["true_label_binary"].fillna(0).astype(int)
    gt["true_label_code"] = gt["true_label_code"].fillna("00")
    return gt


def load_ground_truth(
    gt_csv: Optional[Path],
    submission: pd.DataFrame,
    submission_csv: Optional[Path] = None,
) -> pd.DataFrame:
    """Load or auto-derive ground truth aligned with submission sample_ids."""
    if gt_csv is not None and gt_csv.exists():
        gt = pd.read_csv(gt_csv)
    else:
        logger.info(
            "No ground-truth CSV supplied – deriving from labels.csv + split.json"
        )
        gt = _build_ground_truth_from_internal(submission, submission_csv)
        if gt is None:
            logger.error(
                "Cannot determine ground truth.  "
                "Supply --ground-truth or ensure internal test predictions are present."
            )
            sys.exit(1)

    required = {"sample_id", "true_label_binary", "true_label_code"}
    missing = required - set(gt.columns)
    if missing:
        # Try lenient column mapping
        alt_map = {
            "label": "true_label_binary",
            "defect_label": "true_label_binary",
            "code":  "true_label_code",
            "label_code": "true_label_code",
            "true_code": "true_label_code",
        }
        gt = gt.rename(columns={k: v for k, v in alt_map.items() if k in gt.columns})
        missing = required - set(gt.columns)
        if missing:
            raise ValueError(f"Ground-truth CSV missing columns: {missing}")

    return gt


# ── Calibration plot (text) ───────────────────────────────────────────────────

def _calibration_summary(y_true, p_pred, n_bins=10) -> str:
    bins   = np.linspace(0, 1, n_bins + 1)
    lines  = [f"{'Bin':>12}  {'Conf':>8}  {'Acc':>8}  {'Count':>6}"]
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (p_pred >= lo) & (p_pred < hi)
        if mask.sum() == 0:
            continue
        conf = p_pred[mask].mean()
        acc  = y_true[mask].mean()
        lines.append(f"[{lo:.1f}, {hi:.1f})  {conf:8.4f}  {acc:8.4f}  {mask.sum():6d}")
    return "\n".join(lines)


# ── Main evaluation ───────────────────────────────────────────────────────────

def evaluate(
    submission_csv: Path = SUBMISSION_DEFAULT,
    gt_csv: Optional[Path] = None,
    save_report: bool = True,
) -> dict:
    """Run full evaluation. Returns dict with all metric values."""

    # 1. Load submission  (pred_label_code must stay as string, e.g. "00" not 0)
    sub = pd.read_csv(submission_csv, dtype={"pred_label_code": str})
    # Zero-pad any numeric codes read without leading zero
    sub["pred_label_code"] = sub["pred_label_code"].str.zfill(2)
    logger.info("Loaded submission: %d rows from %s", len(sub), submission_csv)

    # 2. Load ground truth
    gt = load_ground_truth(gt_csv, sub, submission_csv)

    # 3. Merge
    df = sub.merge(gt, on="sample_id", how="inner")
    n_matched = len(df)
    logger.info("Matched %d / %d submission rows with ground truth.", n_matched, len(sub))
    if n_matched == 0:
        logger.error("No rows matched. Check sample_id alignment.")
        sys.exit(1)

    # ── Extract arrays ────────────────────────────────────────────────────────
    y_bin     = df["true_label_binary"].astype(int).to_numpy()
    pred_code = df["pred_label_code"].astype(str).to_numpy()
    true_code = df["true_label_code"].astype(str).to_numpy()
    p_defect  = df["p_defect"].to_numpy(dtype=float)

    # Binary prediction from code
    y_pred_bin = (pred_code != "00").astype(int)

    # ─────────────────────────────────────────────────────────────────────────
    # A. Binary metrics
    # ─────────────────────────────────────────────────────────────────────────
    tn, fp, fn, tp = confusion_matrix(y_bin, y_pred_bin, labels=[0, 1]).ravel()

    binary_f1       = f1_score(y_bin, y_pred_bin, pos_label=1, zero_division=0)
    binary_prec     = precision_score(y_bin, y_pred_bin, pos_label=1, zero_division=0)
    binary_rec      = recall_score(y_bin, y_pred_bin, pos_label=1, zero_division=0)
    binary_acc      = accuracy_score(y_bin, y_pred_bin)
    specificity     = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    roc_auc         = roc_auc_score(y_bin, p_defect) if len(np.unique(y_bin)) > 1 else float("nan")
    pr_auc          = average_precision_score(y_bin, p_defect) if len(np.unique(y_bin)) > 1 else float("nan")

    # ─────────────────────────────────────────────────────────────────────────
    # B. Multi-class defect-type metrics
    # ─────────────────────────────────────────────────────────────────────────
    # Only evaluate type on samples where we have a known true type
    type_mask = ~pd.isna(df["true_label_code"])
    if type_mask.sum() >= 2:
        t_true  = true_code[type_mask]
        t_pred  = pred_code[type_mask]
        present_codes = sorted(set(t_true.tolist()) | set(t_pred.tolist()))

        type_macro_f1  = f1_score(t_true, t_pred, average="macro",    zero_division=0, labels=present_codes)
        type_weighted_f1 = f1_score(t_true, t_pred, average="weighted", zero_division=0, labels=present_codes)
        type_per_class = classification_report(
            t_true, t_pred, labels=present_codes, zero_division=0, output_dict=False
        )
    else:
        type_macro_f1    = float("nan")
        type_weighted_f1 = float("nan")
        type_per_class   = "Insufficient data for multi-class report."

    # ─────────────────────────────────────────────────────────────────────────
    # C. Calibration
    # ─────────────────────────────────────────────────────────────────────────
    ece         = expected_calibration_error(y_bin.astype(float), p_defect)
    brier       = brier_score_loss(y_bin, p_defect)

    # ─────────────────────────────────────────────────────────────────────────
    # D. Final score
    # ─────────────────────────────────────────────────────────────────────────
    if not np.isnan(type_macro_f1):
        final_score = 0.6 * binary_f1 + 0.4 * type_macro_f1
    else:
        final_score = binary_f1  # fall back to binary if no type labels

    metrics = {
        # binary
        "binary_f1":       binary_f1,
        "binary_precision": binary_prec,
        "binary_recall":   binary_rec,
        "binary_accuracy": binary_acc,
        "specificity":     specificity,
        "roc_auc":         roc_auc,
        "pr_auc":          pr_auc,
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        # multi-class
        "type_macro_f1":    type_macro_f1,
        "type_weighted_f1": type_weighted_f1,
        # calibration
        "ece":              ece,
        "brier_score":      brier,
        # composite
        "final_score":      final_score,
        # counts
        "n_samples":        n_matched,
    }

    # ─────────────────────────────────────────────────────────────────────────
    # Print report
    # ─────────────────────────────────────────────────────────────────────────
    sep = "=" * 62

    report = f"""
{sep}
  COMPETITION EVALUATION REPORT
  Submission : {submission_csv.name}
  Samples    : {n_matched}
{sep}

+--- A. BINARY (defect vs good) -------------------------------------------+
|  F1 (defect)     : {binary_f1:.4f}    <- primary metric
|  Precision       : {binary_prec:.4f}
|  Recall          : {binary_rec:.4f}
|  Specificity     : {specificity:.4f}
|  Accuracy        : {binary_acc:.4f}
|  ROC-AUC         : {roc_auc:.4f}
|  PR-AUC          : {pr_auc:.4f}
|
|  Confusion Matrix:
|    TP={tp:4d}   FP={fp:4d}
|    FN={fn:4d}   TN={tn:4d}
+--------------------------------------------------------------------------+

+--- B. DEFECT TYPE (multi-class) -----------------------------------------+
|  Macro F1        : {type_macro_f1:.4f}    <- Type_MacroF1
|  Weighted F1     : {type_weighted_f1:.4f}
+--------------------------------------------------------------------------+

Per-class report:
{type_per_class}

+--- C. CONFIDENCE CALIBRATION --------------------------------------------+
|  ECE (10-bin)    : {ece:.4f}   (lower is better)
|  Brier Score     : {brier:.4f}   (lower is better)
+--------------------------------------------------------------------------+

Calibration grid (p_defect):
{_calibration_summary(y_bin.astype(float), p_defect)}

+--- D. FINAL COMPOSITE SCORE --------------------------------------------+
|  FinalScore = 0.6 x {binary_f1:.4f} + 0.4 x {type_macro_f1:.4f}
|             = {final_score:.4f}
+--------------------------------------------------------------------------+
"""

    print(report)

    if save_report:
        report_path = OUTPUT_DIR / "evaluation_report.txt"
        report_path.write_text(report, encoding="utf-8")
        logger.info("Report saved → %s", report_path)

        metrics_path = OUTPUT_DIR / "evaluation_metrics.json"
        import json as _json
        with open(metrics_path, "w") as f:
            _json.dump(
                {k: (float(v) if not isinstance(v, (int, str)) else v)
                 for k, v in metrics.items()},
                f, indent=2
            )
        logger.info("Metrics JSON saved → %s", metrics_path)

    return metrics


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate competition submission CSV"
    )
    parser.add_argument(
        "--submission", type=Path, default=SUBMISSION_DEFAULT,
        help=f"Path to submission CSV (default: {SUBMISSION_DEFAULT})"
    )
    parser.add_argument(
        "--ground-truth", type=Path, default=None,
        help="Path to ground-truth CSV with columns: "
             "sample_id, true_label_binary, true_label_code"
    )
    parser.add_argument(
        "--no-save", action="store_true",
        help="Do not save report to disk."
    )
    args = parser.parse_args()

    metrics = evaluate(
        submission_csv=args.submission,
        gt_csv=args.ground_truth,
        save_report=not args.no_save,
    )
    print(f"\nFinalScore = {metrics['final_score']:.4f}")
    return metrics


if __name__ == "__main__":
    main()
