"""
Phase 2 – Evaluation utilities for binary weld-defect classification.

Produces:
  • Core metrics: accuracy, precision, recall, F1, AUC-ROC, AUC-PR
  • Confusion matrix (absolute + normalised)
  • Calibration curve
  • ROC + Precision-Recall curves
  • Error breakdown tables (FP / FN samples with probabilities)
  • Per-threshold sweep table for threshold selection
  • Exportable summary dict / text report
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, cast

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.figure import Figure
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve,
    brier_score_loss,
    log_loss,
)
from sklearn.calibration import calibration_curve

from src.config import LABEL_MAP, OUTPUT_DIR, DASHBOARD_DIR

logger = logging.getLogger(__name__)

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.05)
plt.rcParams.update({"figure.dpi": 120, "savefig.dpi": 150, "figure.figsize": (10, 5)})


# ──────────────────────────────────────────────────────────────
# Core metric computation
# ──────────────────────────────────────────────────────────────

def compute_binary_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute a full suite of binary classification metrics.

    Parameters
    ----------
    y_true : array of {0, 1}
    y_prob : array of floats in [0, 1]  (P(defect))
    threshold : decision boundary

    Returns
    -------
    dict with metric_name → value
    """
    y_pred = (y_prob >= threshold).astype(int)

    metrics: Dict[str, float] = {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "specificity": float(recall_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)) if len(set(y_true)) > 1 else float("nan"),
        "avg_precision": float(average_precision_score(y_true, y_prob)) if len(set(y_true)) > 1 else float("nan"),
        "brier_score": float(brier_score_loss(y_true, y_prob)),
        "log_loss": float(log_loss(y_true, np.clip(y_prob, 1e-7, 1 - 1e-7))),
    }

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    metrics["tn"], metrics["fp"], metrics["fn"], metrics["tp"] = cm.ravel().tolist()

    return metrics


# ──────────────────────────────────────────────────────────────
# Threshold sweep  (used for threshold selection on val set)
# ──────────────────────────────────────────────────────────────

def threshold_sweep(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """Evaluate metrics at many thresholds.  Returns a DataFrame."""
    if thresholds is None:
        thresholds = np.arange(0.05, 1.0, 0.01)

    rows = []
    for t in thresholds:
        m = compute_binary_metrics(y_true, y_prob, threshold=t)
        rows.append(m)
    return pd.DataFrame(rows)


def select_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    strategy: str = "f1",
) -> float:
    """Pick the threshold that maximises a given metric on the validation set.

    Strategies
    ----------
    'f1'         : maximise F1 score  (default – balances precision & recall)
    'youden'     : maximise Youden's J = sensitivity + specificity − 1
    'precision90' : lowest threshold achieving ≥ 90 % precision
    'recall95'   : highest threshold achieving ≥ 95 % recall
    """
    sweep_df = threshold_sweep(y_true, y_prob)

    if strategy == "f1":
        idx = sweep_df["f1"].idxmax()
    elif strategy == "youden":
        sweep_df["youden_j"] = sweep_df["recall"] + sweep_df["specificity"] - 1.0
        idx = sweep_df["youden_j"].idxmax()
    elif strategy == "precision90":
        candidates = sweep_df[sweep_df["precision"] >= 0.90]
        idx = candidates["threshold"].idxmin() if len(candidates) else sweep_df["f1"].idxmax()
    elif strategy == "recall95":
        candidates = sweep_df[sweep_df["recall"] >= 0.95]
        idx = candidates["threshold"].idxmax() if len(candidates) else sweep_df["f1"].idxmax()
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    if idx is None:
        raise ValueError("threshold_sweep produced no valid rows; cannot select a threshold")

    chosen = float(sweep_df["threshold"].at[idx])  # type: ignore[arg-type]
    logger.info("Threshold chosen (strategy=%s): %.3f", strategy, chosen)
    return round(chosen, 4)


# ──────────────────────────────────────────────────────────────
# Error breakdown
# ──────────────────────────────────────────────────────────────

def error_breakdown(
    predictions_df: pd.DataFrame,
    threshold: float = 0.5,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return DataFrames of false-positive and false-negative samples.

    predictions_df must have columns: sample_id, label, p_defect
    """
    df = predictions_df.copy()
    df["pred_defect"] = (df["p_defect"] >= threshold).astype(int)

    fp = df[(df["pred_defect"] == 1) & (df["label"] == 0)].sort_values("p_defect", ascending=False)
    fn = df[(df["pred_defect"] == 0) & (df["label"] == 1)].sort_values("p_defect", ascending=True)

    return fp, fn


# ──────────────────────────────────────────────────────────────
# Plotting helpers
# ──────────────────────────────────────────────────────────────

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: Optional[Path] = None,
) -> Figure:
    """Plot absolute + normalised confusion matrices side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    labels = ["good", "defect"]

    for ax, normalize, title in zip(
        axes, [None, "true"], ["Absolute Counts", "Normalised (by true class)"]
    ):
        norm = cast("Literal['true', 'pred', 'all'] | None", normalize)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1], normalize=norm)
        fmt = "d" if normalize is None else ".2%"
        if normalize is not None:
            cm_display = cm
        else:
            cm_display = cm
        sns.heatmap(
            cm_display, annot=True, fmt=fmt, cmap="Blues",
            xticklabels=labels, yticklabels=labels, ax=ax,
            cbar=False, linewidths=0.5,
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(title)

    fig.suptitle("Confusion Matrix", fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def plot_roc_and_pr(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
    save_path: Optional[Path] = None,
) -> Figure:
    """Side-by-side ROC curve and Precision-Recall curve."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ROC
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_prob)
    auc_val = roc_auc_score(y_true, y_prob)
    axes[0].plot(fpr, tpr, lw=2, label=f"AUC = {auc_val:.3f}")
    axes[0].plot([0, 1], [0, 1], "k--", lw=1, alpha=0.4)
    # Mark operating point
    y_pred_t = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred_t, labels=[0, 1])
    tn, fp_cnt, fn_cnt, tp_cnt = cm.ravel()
    op_fpr = fp_cnt / max(fp_cnt + tn, 1)
    op_tpr = tp_cnt / max(tp_cnt + fn_cnt, 1)
    axes[0].plot(op_fpr, op_tpr, "ro", ms=10, label=f"Threshold={threshold:.2f}")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC Curve")
    axes[0].legend(loc="lower right")
    axes[0].grid(True, alpha=0.3)

    # Precision-Recall
    prec, rec, pr_thresholds = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    axes[1].plot(rec, prec, lw=2, label=f"AP = {ap:.3f}")
    # Mark operating point
    op_prec = tp_cnt / max(tp_cnt + fp_cnt, 1)
    op_rec = tp_cnt / max(tp_cnt + fn_cnt, 1)
    axes[1].plot(op_rec, op_prec, "ro", ms=10, label=f"Threshold={threshold:.2f}")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall Curve")
    axes[1].legend(loc="lower left")
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("Discrimination Curves", fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def plot_calibration(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
    save_path: Optional[Path] = None,
    label: str = "Model",
) -> Figure:
    """Reliability diagram (calibration curve)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Calibration curve
    fraction_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="uniform")
    axes[0].plot(mean_pred, fraction_pos, "s-", lw=2, label=label)
    axes[0].plot([0, 1], [0, 1], "k--", lw=1, alpha=0.4, label="Perfectly calibrated")
    axes[0].set_xlabel("Mean predicted probability")
    axes[0].set_ylabel("Fraction of positives")
    axes[0].set_title("Calibration (Reliability) Curve")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Histogram of predicted probabilities
    axes[1].hist(y_prob[y_true == 0], bins=25, alpha=0.6, label="Good (y=0)", color="#4CAF50")
    axes[1].hist(y_prob[y_true == 1], bins=25, alpha=0.6, label="Defect (y=1)", color="#F44336")
    axes[1].set_xlabel("Predicted P(defect)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Probability Distribution by True Class")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("Calibration Analysis", fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def plot_threshold_sweep(
    sweep_df: pd.DataFrame,
    chosen_threshold: float,
    save_path: Optional[Path] = None,
) -> Figure:
    """Plot F1, precision, recall, and accuracy vs threshold."""
    fig, ax = plt.subplots(figsize=(10, 5))

    for metric, color in [
        ("f1", "#2196F3"),
        ("precision", "#FF9800"),
        ("recall", "#4CAF50"),
        ("accuracy", "#9C27B0"),
    ]:
        ax.plot(sweep_df["threshold"], sweep_df[metric], lw=2, label=metric.capitalize(), color=color)

    ax.axvline(chosen_threshold, color="red", ls="--", lw=2, alpha=0.7,
               label=f"Chosen threshold = {chosen_threshold:.3f}")

    ax.set_xlabel("Threshold")
    ax.set_ylabel("Score")
    ax.set_title("Metric vs Threshold Sweep")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def plot_error_examples(
    fp_df: pd.DataFrame,
    fn_df: pd.DataFrame,
    top_n: int = 10,
    save_path: Optional[Path] = None,
) -> Figure:
    """Horizontal bar chart of worst FP and FN samples by confidence."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # False Positives (good samples predicted defect)
    ax = axes[0]
    fp_show = fp_df.head(top_n).copy()
    if len(fp_show) > 0:
        ax.barh(range(len(fp_show)), fp_show["p_defect"].values, color="#FF9800", edgecolor="white")
        ax.set_yticks(range(len(fp_show)))
        ax.set_yticklabels(fp_show["sample_id"].values, fontsize=8)
        ax.set_xlabel("P(defect)")
        ax.axvline(0.5, color="red", ls="--", alpha=0.5)
        ax.invert_yaxis()
    ax.set_title(f"False Positives ({len(fp_df)} total)")

    # False Negatives (defect samples predicted good)
    ax = axes[1]
    fn_show = fn_df.head(top_n).copy()
    if len(fn_show) > 0:
        ax.barh(range(len(fn_show)), fn_show["p_defect"].values, color="#F44336", edgecolor="white")
        ax.set_yticks(range(len(fn_show)))
        ax.set_yticklabels(fn_show["sample_id"].values, fontsize=8)
        ax.set_xlabel("P(defect)")
        ax.axvline(0.5, color="red", ls="--", alpha=0.5)
        ax.invert_yaxis()
    ax.set_title(f"False Negatives ({len(fn_df)} total)")

    fig.suptitle("Error Analysis – Worst Misclassifications", fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


# ──────────────────────────────────────────────────────────────
# Full evaluation report
# ──────────────────────────────────────────────────────────────

def full_evaluation_report(
    predictions_df: pd.DataFrame,
    threshold: float,
    split_name: str = "test",
    save_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Run all evaluations and generate plots.

    Parameters
    ----------
    predictions_df : DataFrame with columns [sample_id, label, p_defect]
    threshold : fixed decision threshold
    split_name : label for the report (e.g. 'val', 'test')
    save_dir : directory for saved plots (defaults to DASHBOARD_DIR)

    Returns
    -------
    dict with all metrics + paths to saved figures
    """
    if save_dir is None:
        save_dir = DASHBOARD_DIR
    save_dir.mkdir(parents=True, exist_ok=True)

    y_true = np.asarray(predictions_df["label"].values, dtype=int)
    y_prob = np.asarray(predictions_df["p_defect"].values, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)

    # 1. Metrics
    metrics = compute_binary_metrics(y_true, y_prob, threshold=threshold)
    logger.info(
        "[%s] acc=%.3f  prec=%.3f  rec=%.3f  f1=%.3f  auc=%.3f  ap=%.3f  brier=%.4f",
        split_name, metrics["accuracy"], metrics["precision"], metrics["recall"],
        metrics["f1"], metrics["roc_auc"], metrics["avg_precision"], metrics["brier_score"],
    )

    # 2. Plots
    saved_plots: Dict[str, Path] = {}

    p = save_dir / f"phase2_{split_name}_confusion_matrix.png"
    plot_confusion_matrix(y_true, y_pred, save_path=p)
    saved_plots["confusion_matrix"] = p
    plt.close()

    p = save_dir / f"phase2_{split_name}_roc_pr.png"
    plot_roc_and_pr(y_true, y_prob, threshold=threshold, save_path=p)
    saved_plots["roc_pr"] = p
    plt.close()

    p = save_dir / f"phase2_{split_name}_calibration.png"
    plot_calibration(y_true, y_prob, save_path=p)
    saved_plots["calibration"] = p
    plt.close()

    # 3. Threshold sweep
    sweep_df = threshold_sweep(y_true, y_prob)
    p = save_dir / f"phase2_{split_name}_threshold_sweep.png"
    plot_threshold_sweep(sweep_df, threshold, save_path=p)
    saved_plots["threshold_sweep"] = p
    plt.close()

    # 4. Error breakdown
    fp_df, fn_df = error_breakdown(predictions_df, threshold=threshold)
    p = save_dir / f"phase2_{split_name}_error_examples.png"
    plot_error_examples(fp_df, fn_df, save_path=p)
    saved_plots["error_examples"] = p
    plt.close()

    # 5. Text report
    report_lines = [
        f"{'=' * 60}",
        f"  PHASE 2 – BINARY CLASSIFICATION REPORT ({split_name.upper()})",
        f"{'=' * 60}",
        f"  Threshold          : {threshold:.4f}",
        f"  Samples            : {len(y_true)}  (good={int((y_true == 0).sum())}, defect={int((y_true == 1).sum())})",
        f"",
        f"  Accuracy           : {metrics['accuracy']:.4f}",
        f"  Precision          : {metrics['precision']:.4f}",
        f"  Recall (Sensitivity): {metrics['recall']:.4f}",
        f"  Specificity        : {metrics['specificity']:.4f}",
        f"  F1 Score           : {metrics['f1']:.4f}",
        f"  ROC AUC            : {metrics['roc_auc']:.4f}",
        f"  Average Precision  : {metrics['avg_precision']:.4f}",
        f"  Brier Score        : {metrics['brier_score']:.4f}",
        f"  Log Loss           : {metrics['log_loss']:.4f}",
        f"",
        f"  Confusion Matrix:",
        f"    TP={metrics['tp']}  FN={metrics['fn']}",
        f"    FP={metrics['fp']}  TN={metrics['tn']}",
        f"",
        f"  False Positives    : {len(fp_df)}",
        f"  False Negatives    : {len(fn_df)}",
        f"{'=' * 60}",
    ]
    report_text = "\n".join(report_lines)

    return {
        "metrics": metrics,
        "sweep_df": sweep_df,
        "fp_df": fp_df,
        "fn_df": fn_df,
        "saved_plots": saved_plots,
        "report_text": report_text,
    }
