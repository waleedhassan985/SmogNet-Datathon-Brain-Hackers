"""Shared configuration for the SmogNet pipeline."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
OUTPUT_DATA_DIR = DATA_DIR / "outputs"

DEFAULT_ROLLING_WINDOW = 72
DEFAULT_TEST_FRACTION = 0.20
MIN_GROUP_SIZE = 24
EPSILON = 1e-6

CANONICAL_POLLUTANTS = ("pm2_5", "pm10", "no", "no2", "so2", "nh3", "co", "o3")
SEVERITY_ORDER = ("moderate", "high", "severe")

