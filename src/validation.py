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
import numpy as np
import baselines
import features
from config import DATE_COL, ID_COL, TARGET

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


# ---------------------------------------------------------------------------
# Metrics (D22). All in DOLLARS: predictions are converted back to $ before
# scoring, whatever scale a model trains on. Spotter's metric is unknown, so
# we report all four; MAE is primary, MAPE second (our choice, not theirs).
# Written by hand (one line each) so every number is explainable; Step 2
# cross-checks them against scikit-learn.
# ---------------------------------------------------------------------------

def _check(y_true, y_pred) -> tuple[np.ndarray, np.ndarray]:
    """Turn inputs into float arrays and refuse anything that would give a misleading score."""
    t = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)
    if t.shape != p.shape:
        raise ValueError(f"Length mismatch: {t.shape} true vs {p.shape} predicted")
    if not (np.isfinite(t).all() and np.isfinite(p).all()):
        raise ValueError("NaN or infinite values in true or predicted rates")
    if (t <= 0).any():
        raise ValueError("True rates must be > 0 (MAPE divides by them)")
    return t, p


def mae(y_true, y_pred) -> float:
    """Mean absolute error: average dollars off per load."""
    t, p = _check(y_true, y_pred)
    return float(np.mean(np.abs(p - t)))


def mape(y_true, y_pred) -> float:
    """Mean absolute percentage error, in % (sklearn returns a FRACTION; we use %)."""
    t, p = _check(y_true, y_pred)
    return float(np.mean(np.abs(p - t) / t) * 100)


def rmse(y_true, y_pred) -> float:
    """Root mean squared error: like MAE, but big misses count extra."""
    t, p = _check(y_true, y_pred)
    return float(np.sqrt(np.mean((p - t) ** 2)))


def bias(y_true, y_pred) -> float:
    """Mean signed error, predicted minus true. Positive = we over-price on average."""
    t, p = _check(y_true, y_pred)
    return float(np.mean(p - t))


METRICS = {"MAE": mae, "MAPE_%": mape, "RMSE": rmse, "bias": bias}


def score(y_true, y_pred) -> dict:
    """All four metrics at once, as {name: value}."""
    return {name: fn(y_true, y_pred) for name, fn in METRICS.items()}


# ---------------------------------------------------------------------------
# The per-fold pipeline (D23). Every model, baseline or not, goes through this.
# ---------------------------------------------------------------------------

def run_fold(raw: pd.DataFrame, fold: Fold, make_model, drop_errors: bool = True) -> dict:
    """split -> fit_cleaning(train only) -> clean both sides -> learn price table
    from train, drop flagged rows from TRAINING only (D13) -> fit -> predict the
    holdout in dollars -> check -> score on ALL holdout rows.

    make_model is a CLASS (e.g. baselines.SegmentMedianRpm), not a fitted model:
    a fresh model is built for every fold, so nothing learned can leak between folds.
    """
    train_raw, test_raw = time_split(raw, fold)

    learned = features.fit_cleaning(train_raw)              # training rows only
    if learned["n_rows"] != len(train_raw):
        raise RuntimeError("fit_cleaning saw a different number of rows than the training side")
    train = features.clean(train_raw, learned)
    test = features.clean(test_raw, learned)                # same learned values, never re-learned

    table = baselines.fit_price_table(train)                # training prices only
    train_flags = baselines.flag_label_errors(train, table)
    train_fit = train[~train_flags["is_error"].to_numpy()] if drop_errors else train

    model = make_model().fit(train_fit)
    pred = np.asarray(model.predict(test), dtype=float)     # always dollars

    if len(pred) != len(test):
        raise ValueError(f"{model.name}: {len(pred)} predictions for {len(test)} rows")
    if not np.isfinite(pred).all() or (pred <= 0).any():
        raise ValueError(f"{model.name}: predictions must be finite and > 0")

    holdout = test.copy()
    holdout["pred"] = pred
    # REPORTING ONLY (the labelled clean-only score in Step 5): flag holdout rows
    # with the TRAINING table. Never feeds back into training or prediction.
    holdout["is_label_error"] = baselines.flag_label_errors(test, table)["is_error"].to_numpy()

    return {
        "fold": fold.name,
        "model": model.name,
        "drop_errors": drop_errors,
        "cleaning_rows": learned["n_rows"],
        "train_last_date": train[DATE_COL].max().date(),
        "dropped": int(train_flags["is_error"].sum()) if drop_errors else 0,
        "fit_rows": len(train_fit),
        "test_rows": len(test),
        "scores": score(holdout[TARGET], pred),             # ALL holdout rows (D13)
        "holdout": holdout,
    }