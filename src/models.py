"""
models.py: Phase 4 models. Same contract as baselines.py (D23):
    fit(train_df) -> self        # CLEANED training rows, label errors removed
    predict(df)   -> np.ndarray  # DOLLARS, one per row, same order
predict() may receive rows that still contain posted_rate, but NEVER reads it.
Target transforms live inside the model.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

import baselines
import features
from config import DATE_COL, MARKET_WINDOW_DAYS, SEED, TARGET
from features import KNOWN_EQUIPMENT

EQUIPMENT_ORDER = sorted(KNOWN_EQUIPMENT)          # fixed one-hot order (decision 4)
BASE_FEATURES = ["distance", "weight", "pickup_lat", "pickup_lon",
                 "delivery_lat", "delivery_lon", "market_index"]
EXTRA_FEATURES = {"weekday", "circuity", "quote_signal", "market_28d", "market_rel"}
EARTH_RADIUS_MI = 3958.8


def straight_line_miles(df: pd.DataFrame) -> np.ndarray:
    """Great-circle (haversine) distance between pickup and delivery coordinates."""
    lat1, lon1, lat2, lon2 = (np.radians(df[c].to_numpy(dtype=float)) for c in
                              ["pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon"])
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_RADIUS_MI * np.arcsin(np.sqrt(a))


def build_features(df: pd.DataFrame, extra: tuple = (), market_daily: pd.Series | None = None) -> pd.DataFrame:
    """Model inputs from a CLEANED file. Never touches TARGET. Stops on any gap.
    market_daily (date -> mean market_index) is needed only for 'market_28d'."""
    unknown = set(extra) - EXTRA_FEATURES
    if unknown:
        raise ValueError(f"Unknown extra features: {sorted(unknown)}")
    X = df[BASE_FEATURES].astype(float).copy()
    for eq in EQUIPMENT_ORDER:
        X[f"eq_{eq}"] = (df["equipment"] == eq).astype(float)
    if "weekday" in extra:
        X["weekday"] = df[DATE_COL].dt.dayofweek.astype(float)
    if "circuity" in extra:
        X["circuity"] = df["distance"].to_numpy(dtype=float) / straight_line_miles(df)
    if "quote_signal" in extra:
        X["quote_signal"] = df["quote_signal"].astype(float)
    if "market_28d" in extra:
        if market_daily is None:
            raise ValueError("market_28d needs market_daily history")
        X["market_28d"] = features.trailing_market_mean(market_daily, df[DATE_COL], MARKET_WINDOW_DAYS)
    if not np.isfinite(X.to_numpy()).all():
        raise ValueError("Features contain NaN or infinite values")
    if "market_rel" in extra:          # today's market vs its own last 28 days: level removed (decision 16b)
        if market_daily is None:
            raise ValueError("market_rel needs market_daily history")
        X["market_rel"] = (df["market_index"].to_numpy(dtype=float)
                           / features.trailing_market_mean(market_daily, df[DATE_COL], MARKET_WINDOW_DAYS))
    return X


class MarketAdjustedRpm:
    """S1 (simplest option): B1's cell rpm, scaled by ONE market_index slope (OLS on log scale)."""
    name = "S1 B1 x market slope"

    def fit(self, train: pd.DataFrame) -> "MarketAdjustedRpm":
        self.table_ = baselines.fit_price_table(train)
        resid = (np.log(baselines.rpm(train).to_numpy(dtype=float))
                 - np.log(baselines.lookup_rpm(train, self.table_)))
        m = train["market_index"].to_numpy(dtype=float)
        self.market_mean_ = float(m.mean())
        x = m - self.market_mean_
        self.slope_ = float(np.sum(x * resid) / np.sum(x * x))
        self.intercept_ = float(resid.mean())
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        x = df["market_index"].to_numpy(dtype=float) - self.market_mean_
        log_rpm = np.log(baselines.lookup_rpm(df, self.table_)) + self.intercept_ + self.slope_ * x
        return np.exp(log_rpm) * df["distance"].to_numpy(dtype=float)


class HGBRate:
    """HistGradientBoostingRegressor on log(rpm) or log(rate); returns dollars."""

    def __init__(self, target: str = "log_rpm", extra: tuple = (), smearing: bool = False,
                 drop: tuple = (), **settings):
        if target not in ("log_rpm", "log_rate"):
            raise ValueError(f"Unknown target: {target}")
        self.target, self.extra, self.smearing = target, tuple(extra), bool(smearing)
        self.drop = tuple(drop)
        if set(self.drop) - set(BASE_FEATURES):
            raise ValueError(f"Can only drop base features: {self.drop}")
        self.settings = {"random_state": SEED, "early_stopping": False, **settings}
        tag = " ".join(f"{k}={v}" for k, v in sorted(settings.items()))
        self.name = (f"HGB {target}" + (f" -{'-'.join(self.drop)}" if self.drop else "")
                     + (f" +{'+'.join(self.extra)}" if self.extra else "")
                     + (f" {tag}" if tag else "") + (" smear" if self.smearing else ""))
        self.extra_daily_ = None                    # optional history added later (December, Phase 6)

    def _offset(self, df: pd.DataFrame) -> np.ndarray:
        d = np.log(df["distance"].to_numpy(dtype=float))
        return d if self.target == "log_rpm" else np.zeros(len(df))

    def extend_history(self, daily: pd.Series) -> "HGBRate":
        """Add market_index days the model didn't train on (e.g. November from validation,
        so December's 28-day window is complete). Features only, never prices."""
        self.extra_daily_ = daily
        return self

    def _history(self, df: pd.DataFrame) -> pd.Series:
        """Training days + any added days + the days in df itself. If a date appears
        in more than one source (only in a random split), their means are averaged."""
        parts = [self.market_daily_, features.daily_market_index(df)]
        if self.extra_daily_ is not None:
            parts.append(self.extra_daily_)
        return pd.concat(parts).groupby(level=0).mean()

    def fit(self, train: pd.DataFrame) -> "HGBRate":
        self.market_daily_ = features.daily_market_index(train)       # features only
        X = build_features(train, self.extra, self.market_daily_).drop(columns=list(self.drop))
        y = np.log(train[TARGET].to_numpy(dtype=float)) - self._offset(train)
        self.columns_ = list(X.columns)
        self.model_ = HistGradientBoostingRegressor(**self.settings).fit(X, y)
        resid = y - self.model_.predict(X)
        self.train_resid_sd_ = float(resid.std())
        self.smear_ = float(np.mean(np.exp(resid))) if self.smearing else 1.0
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        X = build_features(df, self.extra, self._history(df)).drop(columns=list(self.drop))[self.columns_]
        return np.exp(self.model_.predict(X) + self._offset(df)) * self.smear_