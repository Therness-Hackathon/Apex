import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# Add project root so we can import src
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import SENSOR_COLUMNS, FIXED_SEQ_LEN, LABEL_MAP
from src.trainer import WeldClassifier
from src.feature_engineering import extract_sensor_features, extract_image_features, sensor_to_fixed_tensor

# ── Paths ────────────────────────────────────────────────────────
N_SENSOR_CHANNELS = len(SENSOR_COLUMNS)   # 6
N_FEATURES = 104
MODEL_PATH = Path(__file__).resolve().parent / "weld_classifier.pt"
NORM_STATS_PATH = PROJECT_ROOT / "outputs" / "normalize_stats.json"

# ── Point this to the CSV you want to classify ───────────────────
CSV_PATH = Path(
    r"C:\New folder\Apex\sampleData"
    r"\08-17-22-0011-00\08-17-22-0011-00.csv"
)

# ── Load normalisation stats (saved during training) ─────────────
with open(NORM_STATS_PATH) as f:
    raw = json.load(f)
norm_mean = np.array(raw["mean"], dtype=np.float32)
norm_std  = np.array(raw["std"],  dtype=np.float32)
norm_std[norm_std == 0] = 1.0

# ── Read & preprocess the CSV ────────────────────────────────────
df = pd.read_csv(CSV_PATH)
df.columns = df.columns.str.strip()
print(f"CSV loaded: {CSV_PATH.name}  ({len(df)} rows)")

# Branch 1 — Sensor sequence: pad/truncate → z-score → (1, 400, 6)
seq = sensor_to_fixed_tensor(df, FIXED_SEQ_LEN)      # (400, 6) float32
seq = (seq - norm_mean) / norm_std                    # z-score normalise
sensor_tensor = torch.from_numpy(seq).unsqueeze(0)    # (1, 400, 6)

# Branch 2 — Engineered features: stats + image features → (1, 104)
feats = extract_sensor_features(df)                   # dict of ~91 sensor features

# Check for images in the run folder
images_dir = CSV_PATH.parent / "images"
image_paths = sorted(images_dir.glob("*.jpg")) + sorted(images_dir.glob("*.png")) if images_dir.exists() else []
feats.update(extract_image_features(image_paths))     # adds ~13 image features

feat_vals = np.array(list(feats.values()), dtype=np.float32)
feat_vals = np.nan_to_num(feat_vals, nan=0.0)
feat_tensor = torch.from_numpy(feat_vals).unsqueeze(0)  # (1, 104)

print(f"Sensor tensor : {tuple(sensor_tensor.shape)}")
print(f"Feature vector: {tuple(feat_tensor.shape)}  ({len(feats)} features)")

# ── Load model ───────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = WeldClassifier(N_SENSOR_CHANNELS, N_FEATURES).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

sensor_tensor = sensor_tensor.to(device)
feat_tensor   = feat_tensor.to(device)

# ── Inference ────────────────────────────────────────────────────
with torch.no_grad():
    logits = model(sensor_tensor, feat_tensor)

probs = torch.softmax(logits, dim=1)
pred_class = logits.argmax(dim=1).item()
confidence = probs[0, pred_class].item()

print(f"\nLogits:      {logits.cpu().numpy()}")
print(f"Probabilities: good={probs[0,0]:.4f}  defect={probs[0,1]:.4f}")
print(f"Prediction:  {LABEL_MAP[pred_class].upper()} (confidence: {confidence:.2%})")