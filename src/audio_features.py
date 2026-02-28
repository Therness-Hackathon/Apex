"""
Audio feature extraction from .flac weld recordings.

Uses torchaudio to load audio, then extracts:
  - MFCCs (40 coefficients) → mean + std per coeff = 80 features
  - Mel-spectrogram per-band energy → mean + std = 2 * n_mels features
  - Zero-crossing rate → mean + std = 2 features
  - RMS energy → mean + std = 2 features
  - Spectral centroid → mean + std = 2 features
  - Spectral bandwidth → mean + std = 2 features

Total: ~130 scalar audio features per sample.
Returns a dict of {feature_name: float} suitable for merging into feature_table.
Returns all-zeros dict if file is missing or unreadable.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict

import numpy as np

logger = logging.getLogger(__name__)

# ── Feature config ────────────────────────────────────────────────────────────
N_MFCC    = 40
N_MELS    = 64
N_FFT     = 512
HOP_LEN   = 160    # ~10 ms at 16 kHz
TARGET_SR = 16_000  # resample all audio to 16 kHz


def _zero_audio_feats() -> Dict[str, float]:
    """Return all-zero audio feature dict (used when file is missing/broken)."""
    feats: Dict[str, float] = {}
    for i in range(N_MFCC):
        feats[f"aud__mfcc{i:02d}_mean"] = 0.0
        feats[f"aud__mfcc{i:02d}_std"]  = 0.0
    for i in range(N_MELS):
        feats[f"aud__mel{i:02d}_mean"] = 0.0
        feats[f"aud__mel{i:02d}_std"]  = 0.0
    feats["aud__zcr_mean"]       = 0.0
    feats["aud__zcr_std"]        = 0.0
    feats["aud__rms_mean"]       = 0.0
    feats["aud__rms_std"]        = 0.0
    feats["aud__centroid_mean"]  = 0.0
    feats["aud__centroid_std"]   = 0.0
    feats["aud__bandwidth_mean"] = 0.0
    feats["aud__bandwidth_std"]  = 0.0
    feats["aud__duration_s"]     = 0.0
    feats["aud__file_ok"]        = 0.0
    return feats


def extract_audio_features(flac_path: Path) -> Dict[str, float]:
    """Extract scalar audio features from a FLAC file.

    Parameters
    ----------
    flac_path : Path
        Path to the .flac audio file.

    Returns
    -------
    dict : feature_name → float  (all keys always present)
    """
    if not flac_path.exists():
        return _zero_audio_feats()

    try:
        import soundfile as sf
        import torchaudio.transforms as T
        import torch

        # Use soundfile directly — more reliable cross-platform than torchaudio backends
        wave_np_raw, sr = sf.read(str(flac_path), dtype="float32")  # (N,) or (N, C)
        if wave_np_raw.ndim > 1:
            wave_np_raw = wave_np_raw.mean(axis=1)   # → (N,)
        waveform = torch.from_numpy(wave_np_raw).unsqueeze(0)  # (1, N)

        # Resample to TARGET_SR
        if sr != TARGET_SR:
            resampler = T.Resample(orig_freq=sr, new_freq=TARGET_SR)
            waveform = resampler(waveform)
            sr = TARGET_SR

        duration_s = waveform.shape[-1] / sr
        wave_np = waveform.squeeze(0).numpy()  # (N,)

        feats: Dict[str, float] = {}
        feats["aud__duration_s"] = float(duration_s)
        feats["aud__file_ok"]    = 1.0

        # ── MFCCs ────────────────────────────────────────────────────────────
        mfcc_transform = T.MFCC(
            sample_rate=TARGET_SR,
            n_mfcc=N_MFCC,
            melkwargs={"n_fft": N_FFT, "hop_length": HOP_LEN, "n_mels": N_MELS},
        )
        mfcc = mfcc_transform(waveform).squeeze(0).numpy()  # (N_MFCC, T)
        for i in range(N_MFCC):
            feats[f"aud__mfcc{i:02d}_mean"] = float(np.mean(mfcc[i]))
            feats[f"aud__mfcc{i:02d}_std"]  = float(np.std(mfcc[i]))

        # ── Mel-spectrogram per-band energy ──────────────────────────────────
        mel_transform = T.MelSpectrogram(
            sample_rate=TARGET_SR,
            n_fft=N_FFT,
            hop_length=HOP_LEN,
            n_mels=N_MELS,
        )
        mel = mel_transform(waveform).squeeze(0).numpy()  # (N_MELS, T)
        # Power to dB
        mel_db = 10.0 * np.log10(mel + 1e-9)
        for i in range(N_MELS):
            feats[f"aud__mel{i:02d}_mean"] = float(np.mean(mel_db[i]))
            feats[f"aud__mel{i:02d}_std"]  = float(np.std(mel_db[i]))

        # ── Zero-crossing rate ───────────────────────────────────────────────
        zcr_transform = T.Vad(sample_rate=TARGET_SR)  # not available; do it manually
        signs = np.sign(wave_np)
        zcr_frames = _frame_stat(np.abs(np.diff(signs)) / 2, N_FFT, HOP_LEN)
        feats["aud__zcr_mean"] = float(np.mean(zcr_frames))
        feats["aud__zcr_std"]  = float(np.std(zcr_frames))

        # ── RMS energy ───────────────────────────────────────────────────────
        rms_frames = _frame_stat(wave_np ** 2, N_FFT, HOP_LEN, fn="rms")
        feats["aud__rms_mean"] = float(np.mean(rms_frames))
        feats["aud__rms_std"]  = float(np.std(rms_frames))

        # ── Spectral centroid & bandwidth (via FFT) ───────────────────────────
        centroids, bandwidths = _spectral_features(wave_np, TARGET_SR, N_FFT, HOP_LEN)
        feats["aud__centroid_mean"]  = float(np.mean(centroids))
        feats["aud__centroid_std"]   = float(np.std(centroids))
        feats["aud__bandwidth_mean"] = float(np.mean(bandwidths))
        feats["aud__bandwidth_std"]  = float(np.std(bandwidths))

        return feats

    except Exception as exc:
        logger.warning("Audio feature extraction failed for %s: %s", flac_path, exc)
        return _zero_audio_feats()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _frame_stat(
    x: np.ndarray,
    frame_len: int,
    hop: int,
    fn: str = "mean",
) -> np.ndarray:
    """Compute per-frame mean (or rms) of signal x."""
    n_frames = max(1, (len(x) - frame_len) // hop + 1)
    vals = np.zeros(n_frames)
    for i in range(n_frames):
        seg = x[i * hop : i * hop + frame_len]
        if fn == "rms":
            vals[i] = float(np.sqrt(np.mean(seg)))
        else:
            vals[i] = float(np.mean(np.abs(seg)))
    return vals


def _spectral_features(
    wave: np.ndarray,
    sr: int,
    n_fft: int,
    hop: int,
) -> tuple:
    """Return per-frame spectral centroid and bandwidth arrays."""
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    n_frames = max(1, (len(wave) - n_fft) // hop + 1)
    centroids = np.zeros(n_frames)
    bandwidths = np.zeros(n_frames)
    for i in range(n_frames):
        seg = wave[i * hop : i * hop + n_fft]
        if len(seg) < n_fft:
            seg = np.pad(seg, (0, n_fft - len(seg)))
        mag = np.abs(np.fft.rfft(seg * np.hanning(n_fft)))
        mag_sum = mag.sum() + 1e-12
        c = float(np.dot(freqs, mag) / mag_sum)
        centroids[i] = c
        bandwidths[i] = float(np.sqrt(np.dot((freqs - c) ** 2, mag) / mag_sum))
    return centroids, bandwidths
