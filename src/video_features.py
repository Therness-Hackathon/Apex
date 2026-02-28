"""
Video feature extraction from .avi weld recordings.

Uses torchvision to load a fixed number of evenly-spaced frames, then extracts:
  - Per-frame: brightness mean/std, edge density, R/G/B channel means
  - Temporal: frame-to-frame difference (motion / arc flash proxy)
  - Aggregate stats over all sampled frames

Total: ~55 scalar video features per sample.
Returns all-zeros dict if file is missing or unreadable.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List

import numpy as np

logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
N_FRAMES   = 12     # frames to sample per video
FRAME_H    = 128    # resize target
FRAME_W    = 128


def _zero_video_feats() -> Dict[str, float]:
    """Return all-zero video features (used when file is missing or broken)."""
    feats: Dict[str, float] = {}
    for stat in ("mean", "std", "min", "max"):
        feats[f"vid__brightness_{stat}"] = 0.0
        feats[f"vid__edge_{stat}"]        = 0.0
        feats[f"vid__r_channel_{stat}"]   = 0.0
        feats[f"vid__g_channel_{stat}"]   = 0.0
        feats[f"vid__b_channel_{stat}"]   = 0.0
        feats[f"vid__motion_{stat}"]      = 0.0
    feats["vid__n_frames_sampled"] = 0.0
    feats["vid__brightness_trend"] = 0.0
    feats["vid__motion_peak"]      = 0.0
    feats["vid__file_ok"]          = 0.0
    return feats


def extract_video_features(avi_path: Path) -> Dict[str, float]:
    """Extract scalar video features from an AVI file.

    Parameters
    ----------
    avi_path : Path
        Path to the .avi video file.

    Returns
    -------
    dict : feature_name → float  (all keys always present)
    """
    if not avi_path.exists():
        return _zero_video_feats()

    try:
        import torchvision.io as tvio
        import torch
        import torchvision.transforms.functional as TF

        # Read video — returns (T, H, W, C) uint8 tensor
        frames_tensor, _, info = tvio.read_video(str(avi_path), output_format="THWC", pts_unit="sec")

        n_total = frames_tensor.shape[0]
        if n_total == 0:
            return _zero_video_feats()

        # Sample N_FRAMES evenly
        indices = np.linspace(0, n_total - 1, min(N_FRAMES, n_total), dtype=int)
        sampled = frames_tensor[indices]  # (k, H, W, 3) uint8

        # Resize to FRAME_H × FRAME_W: iterate frame by frame
        resized: List[np.ndarray] = []
        for i in range(sampled.shape[0]):
            fr = sampled[i]  # (H, W, 3)
            # Convert to (C, H, W) for TF
            fr_chw = fr.permute(2, 0, 1)
            fr_resized = TF.resize(fr_chw, [FRAME_H, FRAME_W], antialias=True)
            resized.append(fr_resized.numpy().astype(np.float32) / 255.0)

        frames = np.stack(resized, axis=0)  # (k, 3, H, W) float32 [0,1]

        feats: Dict[str, float] = {}
        feats["vid__n_frames_sampled"] = float(len(resized))
        feats["vid__file_ok"]          = 1.0

        # ── Brightness (grayscale luminance) ──────────────────────────────
        # Luminance: 0.299R + 0.587G + 0.114B
        luma = (0.299 * frames[:, 0] + 0.587 * frames[:, 1] + 0.114 * frames[:, 2])
        # luma: (k, H, W)
        frame_brightness = luma.reshape(len(resized), -1).mean(axis=1)  # (k,)
        feats["vid__brightness_mean"] = float(np.mean(frame_brightness))
        feats["vid__brightness_std"]  = float(np.std(frame_brightness))
        feats["vid__brightness_min"]  = float(np.min(frame_brightness))
        feats["vid__brightness_max"]  = float(np.max(frame_brightness))
        # Trend: slope of brightness over time (arc flash detection)
        if len(frame_brightness) > 1:
            x = np.arange(len(frame_brightness), dtype=np.float32)
            feats["vid__brightness_trend"] = float(np.polyfit(x, frame_brightness, 1)[0])
        else:
            feats["vid__brightness_trend"] = 0.0

        # ── Edge density (Sobel-like gradient magnitude) ──────────────────
        edge_densities = []
        for i in range(len(resized)):
            gray = luma[i]  # (H, W)
            gx = np.abs(np.diff(gray, axis=1)).mean()
            gy = np.abs(np.diff(gray, axis=0)).mean()
            edge_densities.append(gx + gy)
        edge_densities = np.array(edge_densities)
        feats["vid__edge_mean"] = float(np.mean(edge_densities))
        feats["vid__edge_std"]  = float(np.std(edge_densities))
        feats["vid__edge_min"]  = float(np.min(edge_densities))
        feats["vid__edge_max"]  = float(np.max(edge_densities))

        # ── Per-channel means ─────────────────────────────────────────────
        for ci, ch in enumerate(("r", "g", "b")):
            ch_means = frames[:, ci].reshape(len(resized), -1).mean(axis=1)
            feats[f"vid__{ch}_channel_mean"] = float(np.mean(ch_means))
            feats[f"vid__{ch}_channel_std"]  = float(np.std(ch_means))
            feats[f"vid__{ch}_channel_min"]  = float(np.min(ch_means))
            feats[f"vid__{ch}_channel_max"]  = float(np.max(ch_means))

        # ── Motion (frame-to-frame absolute difference) ───────────────────
        motion = []
        for i in range(1, len(resized)):
            diff = np.abs(frames[i] - frames[i - 1]).mean()
            motion.append(float(diff))
        if motion:
            motion = np.array(motion)
            feats["vid__motion_mean"] = float(np.mean(motion))
            feats["vid__motion_std"]  = float(np.std(motion))
            feats["vid__motion_min"]  = float(np.min(motion))
            feats["vid__motion_max"]  = float(np.max(motion))
            feats["vid__motion_peak"] = float(np.max(motion))
        else:
            feats["vid__motion_mean"] = 0.0
            feats["vid__motion_std"]  = 0.0
            feats["vid__motion_min"]  = 0.0
            feats["vid__motion_max"]  = 0.0
            feats["vid__motion_peak"] = 0.0

        return feats

    except Exception as exc:
        logger.warning("Video feature extraction failed for %s: %s", avi_path, exc)
        return _zero_video_feats()
