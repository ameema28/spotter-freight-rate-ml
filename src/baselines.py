"""
baselines.py: the expected-price table, the label-error flag (D13) and
baselines B0/B1 (D24).

Everything here uses PRICES, so it lives on the training side and is only
ever fitted on training rows. features.py never sees prices (D18).

Every model in this project follows one interface:
    model.fit(train_df)  -> learns from training rows only, returns itself
    model.predict(df)    -> numpy array of DOLLARS, one per row, same order
Returning plain arrays (not Series) avoids pandas silently matching rows by
index label instead of by position.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import LABEL_ERROR_HIGH, LABEL_ERROR_LOW, TARGET
from features import distance_band


def rpm(df: pd.DataFrame) -> pd.Series:
    """Rate per mile. Needs prices, so training rows only."""
    return df[TARGET] / df["distance"]


def _segment_keys(df: pd.DataFrame) -> list[np.ndarray]:
    """(equipment, band) for every row, e.g. ('Dry Van', '(250, 500]').
    Bands become text so they're easy to print, look up and store."""
    band = distance_band(df["distance"]).astype(str).to_numpy()
    return [df["equipment"].to_numpy(), band]


def fit_price_table(train: pd.DataFrame) -> pd.Series:
    """Median rpm per equipment x distance band, learned from THESE rows only.
    Median, not mean, so the label errors we're hunting barely move it."""
    table = rpm(train).groupby(_segment_keys(train)).median()
    table.index.names = ["equipment", "band"]
    return table


def lookup_rpm(df: pd.DataFrame, table: pd.Series) -> np.ndarray:
    """Expected rpm for every row of df. Uses only equipment and distance
    (no prices), so it works on holdout and validation rows too.
    Stops loudly if a row's segment isn't in the table."""
    keys = pd.MultiIndex.from_arrays(_segment_keys(df))
    values = table.reindex(keys).to_numpy(dtype=float)
    if np.isnan(values).any():
        missing = sorted(set(keys[np.isnan(values)]))
        raise ValueError(f"No price-table entry for: {missing}")
    return values


def flag_label_errors(train: pd.DataFrame, table: pd.Series) -> pd.DataFrame:
    """D13: compare each load's rpm with the expected rpm for its segment."""
    factor = rpm(train).to_numpy() / lookup_rpm(train, table)
    low, high = factor < LABEL_ERROR_LOW, factor > LABEL_ERROR_HIGH
    return pd.DataFrame({"factor": factor, "low": low, "high": high,
                         "is_error": low | high}, index=train.index)


def drop_label_errors(train: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Learn the table from this training fold, flag, and drop the flagged rows.
    Returns (kept_rows, flags) so the caller can report what was dropped.
    Used on TRAINING rows only; holdouts are always scored on all rows."""
    flags = flag_label_errors(train, fit_price_table(train))
    return train[~flags["is_error"].to_numpy()].copy(), flags


class GlobalMedianRate:
    """B0: predict the training median rate for every load. The 'do nothing' floor."""
    name = "B0 global median rate"

    def fit(self, train: pd.DataFrame) -> "GlobalMedianRate":
        self.median_ = float(train[TARGET].median())
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return np.full(len(df), self.median_, dtype=float)


class SegmentMedianRpm:
    """B1: median rpm for the load's equipment x band, times its distance.
    Ignores date, market and location, so it's a fair 'simple rules' bar."""
    name = "B1 median rpm by equipment x band"

    def fit(self, train: pd.DataFrame) -> "SegmentMedianRpm":
        self.table_ = fit_price_table(train)
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return lookup_rpm(df, self.table_) * df["distance"].to_numpy(dtype=float)