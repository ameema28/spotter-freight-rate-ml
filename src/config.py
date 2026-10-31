"""
config.py: the ONE place for paths and settings.
Every other script imports from here, so changing a path or the seed
happens in one spot instead of being scattered across files.
"""
from pathlib import Path

# Project root = the folder that contains src/.
# Path(__file__) is this file; .parent is src/; .parent again is the root.
# Using this (not a hard-coded "E:\Projects\...") means the code works on
# anyone's computer, which matters because Spotter will run it on theirs.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

# Input files (names match Spotter's README)
TRAIN_PATH = DATA_DIR / "train_test.csv"
VALID_PATH = DATA_DIR / "validation.csv"
TEMPLATE_PATH = DATA_DIR / "validation_predictions_template.csv"
DECEMBER_PATH = DATA_DIR / "december_chart_inputs.csv"

# Final deliverable
PREDICTIONS_PATH = PROJECT_ROOT / "validation_predictions.csv"
DECEMBER_OUTPUT_PATH = OUTPUT_DIR / "december_chart_inputs.csv"   # filled copy; data/ original never modified

# Column names used across scripts
TARGET = "posted_rate"
DATE_COL = "date"
ID_COL = "load_id"

# Distance bands for the expected-price table (D13).
# pd.cut includes each band's RIGHT edge: (0, 250], (250, 500], ...
# Max training distance is 3,439.8 mi, so 4,000 covers everything seen so far.
DISTANCE_BAND_EDGES = [0, 250, 500, 1000, 2000, 4000]

# Reproducibility: pass random_state=SEED to every model/sampler.
SEED = 42