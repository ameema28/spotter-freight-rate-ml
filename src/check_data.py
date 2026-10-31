"""
check_data.py: Phase 1 sanity check.
Loads every input CSV and compares it with what the project brief says.
Prints MISMATCH for anything unexpected. It only reads; it changes nothing.
Run from the project root:   python src/check_data.py
"""
import pandas as pd
from config import TRAIN_PATH, VALID_PATH, TEMPLATE_PATH, DECEMBER_PATH

# What the brief tells us to expect. If the real files differ, we want to
# find out NOW, before any modelling decision is built on a wrong assumption.
FEATURES = ["load_id", "pickup", "delivery", "pickup_lat", "pickup_lon",
            "delivery_lat", "delivery_lon", "distance", "equipment",
            "weight", "date", "market_index", "quote_signal"]

EXPECTED = {
    "train_test": {"path": TRAIN_PATH, "rows": 48_000,
                   "columns": FEATURES + ["posted_rate"],
                   "date_range": ("2025-01-01", "2025-10-31")},
    "validation": {"path": VALID_PATH, "rows": 12_000,
                   "columns": FEATURES,
                   "date_range": ("2025-11-01", "2025-12-31")},
    "template":   {"path": TEMPLATE_PATH, "rows": 12_000,
                   "columns": ["load_id", "predicted_rate"],
                   "date_range": None},          # template has no date column
    "december":   {"path": DECEMBER_PATH, "rows": 31,
                   "columns": ["pickup", "delivery", "distance", "equipment",
                               "weight", "date", "predicted_rate"],
                   "date_range": ("2025-12-01", "2025-12-31")},
}

problems = 0

def report(ok: bool, message: str) -> None:
    """Print OK or MISMATCH and count the mismatches."""
    global problems
    if ok:
        print(f"  OK        {message}")
    else:
        problems += 1
        print(f"  MISMATCH  {message}")

frames = {}
for name, exp in EXPECTED.items():
    print(f"\n=== {name}  ({exp['path'].name}) ===")
    df = pd.read_csv(exp["path"])
    frames[name] = df

    report(len(df) == exp["rows"], f"rows: {len(df):,} (expected {exp['rows']:,})")
    same_cols = list(df.columns) == exp["columns"]
    report(same_cols, "columns match expected names and order")
    if not same_cols:
        print("    got:", list(df.columns))

    print("  dtypes:")
    for col, dtype in df.dtypes.items():
        print(f"    {col:<15} {dtype}")

    if exp["date_range"] is not None:
        # errors="coerce" turns unreadable dates into NaT so we can count them
        dates = pd.to_datetime(df["date"], errors="coerce")
        bad = int(dates.isna().sum())
        report(bad == 0, f"unparseable dates: {bad}")
        lo, hi = str(dates.min().date()), str(dates.max().date())
        report((lo, hi) == exp["date_range"],
               f"date range: {lo} to {hi} (expected {exp['date_range'][0]} to {exp['date_range'][1]})")

# Checks that compare files with each other
print("\n=== cross-file checks ===")
train_ids = frames["train_test"]["load_id"]
valid_ids = frames["validation"]["load_id"]
templ_ids = frames["template"]["load_id"]
report(train_ids.is_unique, "train_test load_ids are unique")
report(valid_ids.is_unique, "validation load_ids are unique")
report(set(valid_ids) == set(templ_ids), "template has exactly the same load_ids as validation")
overlap = len(set(train_ids) & set(valid_ids))
report(overlap == 0, f"load_ids shared between train_test and validation: {overlap}")

print(f"\nDONE: {problems} mismatch(es).")