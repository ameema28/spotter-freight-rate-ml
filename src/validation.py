"""
validation.py: time-based splits (D20, D21), metrics and the per-fold
pipeline (added in later steps).

A fold = "train on these dates, test on those dates". Splits work on RAW
rows; cleaning happens AFTER the split (fit_cleaning on the train side only),
so no held-out row can influence anything learned.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from config import DATE_COL, ID_COL


@dataclass(frozen=True)          # frozen = the boundaries can't be changed by accident
class Fold:
    name: str
    train_start: str             # all dates inclusive
    train_end: str
    test_start: str
    test_end: str


# D20: main holdout. 2-month horizon in the low market regime, like Nov-Dec.
# Looked at only at checkpoints (baseline, chosen model, final model).
MAIN_FOLD = Fold("main", "2025-01-01", "2025-08-31", "2025-09-01", "2025-10-31")

# D21: decision fold for experiments. Its test month (Aug) sits inside MAIN's
# training side, and it never touches Sep-Oct, so experiments can't leak
# into the main holdout.
DECISION_FOLD = Fold("decision", "2025-01-01", "2025-07-31", "2025-08-01", "2025-08-31")


def time_split(df: pd.DataFrame, fold: Fold) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (train_rows, test_rows) as COPIES of the raw rows for this fold.
    Rows outside both windows (e.g. Sep-Oct for the decision fold) are left out."""
    train_end, test_start = pd.Timestamp(fold.train_end), pd.Timestamp(fold.test_start)
    if train_end >= test_start:
        raise ValueError(f"Fold '{fold.name}': training must end before testing starts")

    dates = pd.to_datetime(df[DATE_COL], errors="raise")   # parsed copy; df is untouched
    # Guard: dates must be whole days, otherwise an inclusive end date like
    # 2025-08-31 would silently drop a load at 2025-08-31 10:00.
    if not (dates == dates.dt.normalize()).all():
        raise ValueError("Dates contain times of day; boundaries assume whole days")

    in_train = dates.between(pd.Timestamp(fold.train_start), train_end)    # inclusive both ends
    in_test = dates.between(test_start, pd.Timestamp(fold.test_end))

    train, test = df[in_train].copy(), df[in_test].copy()

    # Belt and braces: no load may appear on both sides.
    shared = set(train[ID_COL]) & set(test[ID_COL])
    if shared:
        raise ValueError(f"Fold '{fold.name}': {len(shared)} load_id(s) on both sides")
    return train, test


def split_summary(train: pd.DataFrame, test: pd.DataFrame, fold: Fold) -> pd.DataFrame:
    """One row per side: row count and first/last date. This is the evidence for the report."""
    rows = []
    for side, part in [("train", train), ("test", test)]:
        d = pd.to_datetime(part[DATE_COL])
        rows.append({"fold": fold.name, "side": side, "rows": len(part),
                     "first": d.min().date(), "last": d.max().date()})
    return pd.DataFrame(rows)