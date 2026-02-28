# Apex Weld Quality – Data Card

## Dataset Overview
| Property | Value |
|----------|-------|
| Dataset name | Apex Weld Quality – sampleData |
| Date range | Aug 17–18, 2022 |
| Total weld runs | 10 |
| Labelled runs | 10 (good=7, defect=3) — **placeholder labels** |
| Sensor channels | 6 (Pressure, CO2 Weld Flow, Feed, Primary Weld Current, Wire Consumed, Secondary Weld Voltage) |
| Sampling rate | ~9–10 Hz (median interval ≈ 110 ms) |
| Rows per run | ~340 |
| Weld duration | ~37 s per run |
| Images per run | 5 JPGs (post-weld inspection photos) |

## Label Definitions
- **0 = good**: Normal weld with no identified defects
- **1 = defect**: Weld with one or more quality issues

> **Warning:** The labels in `labels.csv` are **placeholders**. Replace them with
> ground-truth annotations before training.

## Sensor Columns and Units
| Column | Description |
|--------|-------------|
| Pressure | Chamber pressure (likely PSI or bar) |
| CO2 Weld Flow | Shielding-gas flow rate |
| Feed | Wire feed speed |
| Primary Weld Current | Arc current (amps) |
| Wire Consumed | Cumulative wire consumed (mm or in) |
| Secondary Weld Voltage | Arc voltage (volts) |

## Preprocessing Choices
| Choice | Value |
|--------|-------|
| Unit of prediction | One complete weld run (Part No) |
| Sequence length | Fixed at 350 rows (zero-pad if shorter, truncate if longer) |
| Normalization | Per-channel z-score (mean/std computed from train set only) |
| Image processing | Grayscale, resized to 224×224, basic pixel statistics |
| Split strategy | Group-by-date prefix to prevent temporal leakage |
| Split ratios | Train 60% / Val 20% / Test 20% |
| Random seed | 42 |

## Feature Engineering
- ~60+ engineered features per run
- **Global stats:** mean, std, min, max, median, range, IQR for each sensor
- **Windowed volatility:** sliding-window std, mean-of-std, std-of-mean
- **Rate of change:** first-difference mean, std, max, min
- **Weld phase:** arc fraction, arc start/end index, duration fraction
- **Image stats:** brightness (mean/std), histogram entropy, edge density

## Known Issues
1. **Placeholder labels** – must be replaced with real annotations.
2. **No audio/video** – only CSVs and still images present; pipeline designed to be extended.
3. **Small dataset** (10 runs) – high overfitting risk; augmentation recommended.
4. **Remarks column** always empty across all runs.

## Files Produced by Phase 1
```
outputs/
  feature_table.csv       – Full feature matrix (samples × features)
  manifest.csv            – Run manifest with validation info
  normalize_stats.json    – Z-score parameters (train-set)
  phase1_summary.txt      – Text summary report
  data_card.md            – This document
  dashboard/              – Analysis plots (PNG)
splits/
  split.json              – Train/val/test split definition
labels.csv                – Label file (sample_id → label)
dashboard.ipynb           – Interactive analysis notebook
```
