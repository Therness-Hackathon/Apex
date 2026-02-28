"""
Central configuration for the Apex weld-quality pipeline.
All paths, hyper-parameters, and column definitions live here so every
module shares a single source of truth.
"""

from pathlib import Path
from typing import Dict, List, Tuple

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
GOOD_WELD_DIR = PROJECT_ROOT / "good_weld"
DEFECT_WELD_DIR = PROJECT_ROOT / "defect-weld"
DATA_DIR = PROJECT_ROOT / "sampleData"      # legacy – kept for compat
LABELS_CSV = PROJECT_ROOT / "labels.csv"
SPLIT_DIR = PROJECT_ROOT / "splits"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
DASHBOARD_DIR = OUTPUT_DIR / "dashboard"
MODEL_DIR = OUTPUT_DIR / "models"

# ──────────────────────────────────────────────
# Sensor CSV schema
# ──────────────────────────────────────────────
SENSOR_COLUMNS: List[str] = [
    "Pressure",
    "CO2 Weld Flow",
    "Feed",
    "Primary Weld Current",
    "Wire Consumed",
    "Secondary Weld Voltage",
]
TIMESTAMP_COLS = ["Date", "Time"]
PART_NO_COL = "Part No"
REMARKS_COL = "Remarks"

# ──────────────────────────────────────────────
# Preprocessing
# ──────────────────────────────────────────────
FIXED_SEQ_LEN = 400          # padded/truncated row count (covers 305-380 range)
IMAGE_SIZE = (224, 224)

# ──────────────────────────────────────────────
# Labels
# ──────────────────────────────────────────────
LABEL_COL = "label"            # 0 = good, 1 = defect
SAMPLE_ID_COL = "sample_id"    # matches the run-folder name
CATEGORY_COL = "category"      # category folder name
DEFECT_TYPE_COL = "defect_type"  # specific defect sub-type

LABEL_MAP: Dict[int, str] = {0: "good", 1: "defect"}
LABEL_INV: Dict[str, int] = {v: k for k, v in LABEL_MAP.items()}

# Defect subtype mapping (category folder prefix → defect name)
DEFECT_TYPES: Dict[str, str] = {
    "burnthrough": "burnthrough",
    "crater_cracks": "crater_cracks",
    "excessive_convexity": "excessive_convexity",
    "excessive_penetration": "excessive_penetration",
    "lack_of_fusion": "lack_of_fusion",
    "overlap": "overlap",
}

# ──────────────────────────────────────────────
# Train / Val / Test split
# ──────────────────────────────────────────────
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_SEED = 42

# ──────────────────────────────────────────────
# Feature engineering – sensor aggregation windows
# ──────────────────────────────────────────────
WINDOW_SIZE = 50
WINDOW_STRIDE = 25

# ──────────────────────────────────────────────
# Training hyper-parameters
# ──────────────────────────────────────────────
BATCH_SIZE = 32
NUM_EPOCHS = 30
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
HIDDEN_DIMS: Tuple[int, ...] = (256, 128, 64)
DROPOUT = 0.3
EARLY_STOP_PATIENCE = 7
