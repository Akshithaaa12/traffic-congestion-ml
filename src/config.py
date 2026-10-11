"""Shared paths and constants for the Traffic Congestion Prediction project."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "traffic.csv"
OUT_DIR = ROOT / "outputs"
FIG_DIR = OUT_DIR / "figures"
MODEL_DIR = ROOT / "models"
for _d in (OUT_DIR, FIG_DIR, MODEL_DIR):
    _d.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

# Chronological split (no shuffling -> no look-ahead leakage)
TRAIN_END = "2017-02-28 23:00:00"   # train: Nov 2015 - Feb 2017
VAL_END = "2017-03-31 23:00:00"     # validation: Mar 2017 (early stopping / model selection)
#                                     test: Apr 2017 - Jun 2017 (never seen during training)

# Capacity proxy: rolling 95th percentile of the PAST 28 days of hourly volume at that junction
CAPACITY_WINDOW_H = 28 * 24
CAPACITY_QUANTILE = 0.95
CAPACITY_MIN_PERIODS = 7 * 24

CLASS_NAMES = ["Low", "Moderate", "High"]
CLASS_COLORS = {"Low": "#2e9d5b", "Moderate": "#f0a202", "High": "#d1495b"}
