"""
Binary weld-quality classifier – training, evaluation, and inference.

Architecture
────────────
Two input branches are fused:
  1. **Sensor branch** – 1-D CNN over the (T, C) time-series
  2. **Feature branch** – MLP on the pre-computed flat features

The outputs are concatenated and passed through a classification head.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.config import (
    BATCH_SIZE,
    NUM_EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    HIDDEN_DIMS,
    DROPOUT,
    EARLY_STOP_PATIENCE,
    MODEL_DIR,
    SENSOR_COLUMNS,
    FIXED_SEQ_LEN,
)

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────

class SensorCNN(nn.Module):
    """1-D Conv stack over time-series (batch, T, C) → (batch, feat_dim)."""

    def __init__(self, in_channels: int, out_dim: int = 128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),

            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Linear(128, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C) → transpose to (B, C, T) for Conv1d
        x = x.transpose(1, 2)
        x = self.conv(x).squeeze(-1)    # (B, 128)
        return self.fc(x)               # (B, out_dim)


class WeldClassifier(nn.Module):
    """Dual-branch binary classifier: sensor CNN + feature MLP → logits."""

    def __init__(
        self,
        n_sensor_channels: int,
        n_features: int,
        hidden_dims: Tuple[int, ...] = HIDDEN_DIMS,
        dropout: float = DROPOUT,
    ):
        super().__init__()
        sensor_out = 128
        feat_out = 64

        # Branch 1: sensor time-series
        self.sensor_branch = SensorCNN(n_sensor_channels, sensor_out)

        # Branch 2: flat features
        self.feat_fc = nn.Sequential(
            nn.Linear(n_features, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[0], feat_out),
            nn.ReLU(inplace=True),
        )

        # Fusion head
        fused_dim = sensor_out + feat_out
        head_layers: List[nn.Module] = []
        in_d = fused_dim
        for h in hidden_dims[1:]:
            head_layers += [
                nn.Linear(in_d, h),
                nn.BatchNorm1d(h),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ]
            in_d = h
        head_layers.append(nn.Linear(in_d, 2))  # 2-class logits
        self.head = nn.Sequential(*head_layers)

    def forward(
        self, sensor_seq: torch.Tensor, features: torch.Tensor
    ) -> torch.Tensor:
        s = self.sensor_branch(sensor_seq)       # (B, 128)
        f = self.feat_fc(features)               # (B, 64)
        fused = torch.cat([s, f], dim=1)         # (B, 192)
        return self.head(fused)                  # (B, 2)


# ──────────────────────────────────────────────
# Training loop
# ──────────────────────────────────────────────

def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train_model(
    train_loader: DataLoader,
    val_loader: DataLoader,
    n_features: int,
    num_epochs: int = NUM_EPOCHS,
    lr: float = LEARNING_RATE,
    weight_decay: float = WEIGHT_DECAY,
    patience: int = EARLY_STOP_PATIENCE,
    class_weights: Optional[torch.Tensor] = None,
) -> Tuple[WeldClassifier, Dict[str, List[float]]]:
    """
    Train the WeldClassifier.

    Returns
    -------
    model : WeldClassifier  (best checkpoint by val loss)
    history : dict with keys 'train_loss', 'val_loss', 'val_acc', 'val_f1'
    """
    device = _get_device()
    logger.info("Training on device: %s", device)

    n_channels = len(SENSOR_COLUMNS)
    model = WeldClassifier(n_channels, n_features).to(device)

    if class_weights is not None:
        class_weights = class_weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=lr * 0.01,
    )

    history: Dict[str, List[float]] = {
        "train_loss": [],
        "val_loss": [],
        "val_acc": [],
        "val_f1": [],
    }

    best_val_loss = float("inf")
    best_state = None
    wait = 0

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()

        # ── train ──
        model.train()
        running_loss = 0.0
        n_train = 0
        for batch in train_loader:
            seq = batch["sensor_seq"].to(device)
            feat = batch["features"].to(device)
            lbl = batch["label"].to(device)

            # Skip unlabelled (label == -1)
            mask = lbl >= 0
            if mask.sum() == 0:
                continue
            seq, feat, lbl = seq[mask], feat[mask], lbl[mask]

            optimizer.zero_grad()
            logits = model(seq, feat)
            loss = criterion(logits, lbl)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            running_loss += loss.item() * lbl.size(0)
            n_train += lbl.size(0)

        avg_train_loss = running_loss / max(n_train, 1)

        # ── val ──
        val_loss, val_acc, val_f1 = evaluate(model, val_loader, criterion, device)

        scheduler.step()
        elapsed = time.time() - t0

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)

        logger.info(
            "Epoch %02d/%d  train_loss=%.4f  val_loss=%.4f  val_acc=%.3f  "
            "val_f1=%.3f  lr=%.1e  (%.1fs)",
            epoch, num_epochs, avg_train_loss, val_loss, val_acc, val_f1,
            optimizer.param_groups[0]["lr"], elapsed,
        )

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                logger.info("Early stopping at epoch %d (patience=%d)", epoch, patience)
                break

    # Restore best weights
    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)

    return model, history


@torch.no_grad()
def evaluate(
    model: WeldClassifier,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, float]:
    """Evaluate model on a DataLoader. Returns (loss, accuracy, f1)."""
    model.eval()
    total_loss = 0.0
    all_preds: List[int] = []
    all_labels: List[int] = []

    for batch in loader:
        seq = batch["sensor_seq"].to(device)
        feat = batch["features"].to(device)
        lbl = batch["label"].to(device)

        mask = lbl >= 0
        if mask.sum() == 0:
            continue
        seq, feat, lbl = seq[mask], feat[mask], lbl[mask]

        logits = model(seq, feat)
        loss = criterion(logits, lbl)
        total_loss += loss.item() * lbl.size(0)

        preds = logits.argmax(dim=1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(lbl.cpu().tolist())

    n = max(len(all_labels), 1)
    avg_loss = total_loss / n

    # Accuracy
    correct = sum(p == l for p, l in zip(all_preds, all_labels))
    acc = correct / n

    # F1 (binary, positive class = 1 = defect)
    tp = sum(p == 1 and l == 1 for p, l in zip(all_preds, all_labels))
    fp = sum(p == 1 and l == 0 for p, l in zip(all_preds, all_labels))
    fn = sum(p == 0 and l == 1 for p, l in zip(all_preds, all_labels))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    return avg_loss, acc, f1


@torch.no_grad()
def predict(
    model: WeldClassifier,
    loader: DataLoader,
    device: Optional[torch.device] = None,
) -> Dict[str, Dict]:
    """Run inference, return {sample_id: {pred, prob_good, prob_defect}}."""
    if device is None:
        device = _get_device()
    model.eval()
    model.to(device)

    results: Dict[str, Dict] = {}
    for batch in loader:
        seq = batch["sensor_seq"].to(device)
        feat = batch["features"].to(device)
        sids = batch["sample_id"]

        logits = model(seq, feat)
        probs = torch.softmax(logits, dim=1)  # (B, 2)
        preds = logits.argmax(dim=1)

        for i, sid in enumerate(sids):
            results[sid] = {
                "pred": int(preds[i].item()),
                "prob_good": float(probs[i, 0].item()),
                "prob_defect": float(probs[i, 1].item()),
            }

    return results


def save_model(model: WeldClassifier, path: Optional[Path] = None) -> Path:
    """Save model weights."""
    if path is None:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        path = MODEL_DIR / "weld_classifier.pt"
    torch.save(model.state_dict(), path)
    logger.info("Model saved to %s", path)
    return path


def compute_class_weights(manifest, label_col: str = "label") -> torch.Tensor:
    """Compute inverse-frequency class weights for imbalanced data."""
    counts = manifest[label_col].value_counts().sort_index()
    total = counts.sum()
    weights = total / (len(counts) * counts.values.astype(np.float64))
    return torch.tensor(weights, dtype=torch.float32)
