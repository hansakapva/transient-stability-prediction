"""Project-wide settings: file locations, feature names, label coding, split sizes, forest grid."""
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT_DIR / "IEEE39_voltage_frequency_4sample_dataset.csv"
MODEL_DIR = PROJECT_DIR / "models"
MODEL_FILE = MODEL_DIR / "stability_rf.joblib"
FIGURE_DIR = PROJECT_DIR / "figures"
REPORT_DIR = PROJECT_DIR / "reports"
EXAMPLE_DIR = PROJECT_DIR / "examples"

# 10 generator buses of the IEEE 39-bus system, 4 time samples of each quantity.
BUSES = [f"B{n}" for n in range(30, 40)]
N_SAMPLES = 4
VOLT_FEATURES = [f"{b}_Vmag_{k}" for b in BUSES for k in range(1, N_SAMPLES + 1)]
FREQ_FEATURES = [f"{b}_FreqHz_{k}" for b in BUSES for k in range(1, N_SAMPLES + 1)]
FEATURES = VOLT_FEATURES + FREQ_FEATURES          # 80 model inputs, in this order

LABEL = "Overall_Stability"      # from the TSI value
GROUP = "Contingency_ID"         # one contingency = 5 rows (one per load level)
STABLE, UNSTABLE = 1, 0
CLASS_NAMES = {STABLE: "Stable", UNSTABLE: "Unstable"}

SEED = 42
TEST_FRACTION = 0.15
VAL_FRACTION = 0.15

# Random forest: number of trees and the grid searched by grouped cross-validation.
N_TREES = 300
RF_GRID = {
    "max_depth": [None, 16, 8],              # None = grow until leaves are pure
    "min_samples_leaf": [1, 3, 10],           # larger = smoother trees, less overfitting
    "max_features": ["sqrt", 0.3, 0.6],       # share of inputs tried at each split (sqrt(80) ~ 9)
    "class_weight": ["balanced", "balanced_subsample"],
}
