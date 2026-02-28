"""
Phase 2 – Temperature Scaling for confidence calibration.

After training, the raw softmax probabilities often over- or under-estimate
true class likelihood.  Temperature scaling learns a single scalar T > 0 on
the **validation set** such that  softmax(logits / T)  is better calibrated.

Reference : Guo et al., "On Calibration of Modern Neural Networks", ICML 2017.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.trainer import WeldClassifier, _get_device

logger = logging.getLogger(__name__)


class TemperatureScaler(nn.Module):
    """Wraps a trained WeldClassifier and learns a temperature parameter."""

    def __init__(self, model: WeldClassifier):
        super().__init__()
        self.model = model
        # Temperature initialised at 1.0 (identity)
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(
        self, sensor_seq: torch.Tensor, features: torch.Tensor
    ) -> torch.Tensor:
        """Return temperature-scaled logits."""
        with torch.no_grad():
            logits = self.model(sensor_seq, features)
        return logits / self.temperature

    def calibrated_probs(
        self, sensor_seq: torch.Tensor, features: torch.Tensor
    ) -> torch.Tensor:
        """Return calibrated softmax probabilities."""
        scaled_logits = self.forward(sensor_seq, features)
        return torch.softmax(scaled_logits, dim=1)


def fit_temperature(
    model: WeldClassifier,
    val_loader: DataLoader,
    max_iter: int = 200,
    lr: float = 0.01,
    device: Optional[torch.device] = None,
) -> TemperatureScaler:
    """Learn optimal temperature on the validation set.

    Uses NLL (CrossEntropy on scaled logits) as the calibration objective.

    Parameters
    ----------
    model : trained WeldClassifier (frozen)
    val_loader : validation DataLoader
    max_iter : optimisation steps
    lr : learning rate for the single temperature parameter

    Returns
    -------
    TemperatureScaler with learned temperature
    """
    if device is None:
        device = _get_device()

    model.eval()
    scaler = TemperatureScaler(model).to(device)

    # Freeze everything except temperature
    for p in scaler.model.parameters():
        p.requires_grad = False
    scaler.temperature.requires_grad = True

    # Collect all validation logits + labels (small enough for memory)
    all_logits = []
    all_labels = []
    with torch.no_grad():
        for batch in val_loader:
            seq = batch["sensor_seq"].to(device)
            feat = batch["features"].to(device)
            lbl = batch["label"].to(device)

            mask = lbl >= 0
            if mask.sum() == 0:
                continue

            logits = model(seq[mask], feat[mask])
            all_logits.append(logits)
            all_labels.append(lbl[mask])

    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.LBFGS([scaler.temperature], lr=lr, max_iter=max_iter)

    # Before calibration
    with torch.no_grad():
        nll_before = criterion(all_logits, all_labels).item()
    logger.info("NLL before temperature scaling: %.4f  (T=1.0)", nll_before)

    def _closure():
        optimizer.zero_grad()
        # Keep temperature strictly positive (prevents NLL-minimising sign flip)
        with torch.no_grad():
            scaler.temperature.clamp_(min=0.05, max=10.0)
        scaled = all_logits / scaler.temperature
        loss = criterion(scaled, all_labels)
        loss.backward()
        return loss

    optimizer.step(_closure)

    # Final clamp after optimisation
    with torch.no_grad():
        scaler.temperature.clamp_(min=0.05, max=10.0)

    T_val = scaler.temperature.item()
    with torch.no_grad():
        nll_after = criterion(all_logits / scaler.temperature, all_labels).item()

    # If calibration did not improve, fall back to T = 1.0
    if nll_after > nll_before:
        logger.warning(
            "Temperature calibration did not improve NLL (%.4f → %.4f). "
            "Resetting T=1.0",
            nll_before, nll_after,
        )
        scaler.temperature.data.fill_(1.0)
        T_val = 1.0
        nll_after = nll_before

    logger.info("NLL after  temperature scaling: %.4f  (T=%.4f)", nll_after, T_val)
    logger.info("Temperature learned: %.4f", T_val)

    return scaler


@torch.no_grad()
def predict_calibrated(
    scaler: TemperatureScaler,
    loader: DataLoader,
    device: Optional[torch.device] = None,
) -> Dict[str, Dict]:
    """Run calibrated inference.

    Returns
    -------
    dict  { sample_id: { pred, p_defect, p_good, confidence } }
    """
    if device is None:
        device = _get_device()
    scaler.eval()
    scaler.to(device)

    results = {}
    for batch in loader:
        seq = batch["sensor_seq"].to(device)
        feat = batch["features"].to(device)
        sids = batch["sample_id"]
        labels = batch["label"]

        probs = scaler.calibrated_probs(seq, feat)  # (B, 2)

        for i, sid in enumerate(sids):
            p_defect = float(probs[i, 1].item())
            p_good = float(probs[i, 0].item())
            pred = 1 if p_defect >= 0.5 else 0  # default; caller can re-threshold
            confidence = max(p_defect, p_good)
            lbl = int(labels[i].item()) if labels[i].item() >= 0 else None

            results[sid] = {
                "p_defect": p_defect,
                "p_good": p_good,
                "pred_defect": pred,
                "confidence": confidence,
                "label": lbl,
            }

    return results
