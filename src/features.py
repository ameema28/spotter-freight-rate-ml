"""
features.py: cleaning rules shared by training AND prediction (D1, D18).

    learned = fit_cleaning(train_df)   # learn values from TRAINING rows only
    cleaned = clean(df, learned)       # apply the same rules to any file

Every rule here is logged in DECISIONS.md (D10-D18).
"""
from __future__ import annotations

from config import DATE_COL, DISTANCE_BAND_EDGES   # one source of truth (housekeeping)

import pandas as pd

KNOWN_EQUIPMENT = {"Dry Van", "Flatbed", "Reefer"}   # D14


def fix_weight_sign(weight: pd.Series) -> pd.Series:
    """D10: negative weights are sign errors, so take the absolute value.
    Missing values stay missing here (abs of NaN is NaN)."""
    return weight.abs()


def distance_band(distance: pd.Series) -> pd.Series:
    """D13: put each distance into a band, e.g. 360 -> (250, 500].

    Uses only distance (a feature, no prices), so it is safe on any file.
    pd.cut returns NaN for a distance outside the edges or a missing one;
    a NaN band would silently break the price table later, so stop loudly.
    """
    bands = pd.cut(distance, DISTANCE_BAND_EDGES)        # right edge included
    if bands.isna().any():
        bad = distance[bands.isna()]
        raise ValueError(
            f"{len(bad)} distance(s) missing or outside "
            f"{DISTANCE_BAND_EDGES[0]}-{DISTANCE_BAND_EDGES[-1]} mi, "
            f"e.g. {bad.head().tolist()}"
        )
    return bands


def city_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    """One row per city with its (lat, lon), taken from both ends of each load."""
    pick = df[["pickup", "pickup_lat", "pickup_lon"]].set_axis(["city", "lat", "lon"], axis=1)
    deli = df[["delivery", "delivery_lat", "delivery_lon"]].set_axis(["city", "lat", "lon"], axis=1)
    coords = pd.concat([pick, deli]).drop_duplicates()
    # Step 5 showed one coordinate pair per city. Stop loudly if that ever breaks.
    if coords["city"].duplicated().any():
        raise ValueError("A city has more than one coordinate pair")
    return coords.set_index("city")


def fit_cleaning(train_df: pd.DataFrame) -> dict:
    """Learn every data-dependent value from TRAINING rows only (D18).

    Called on the full training file for the final model, and on each
    training fold during validation, so held-out rows never influence it.
    Returns a plain dict of numbers and lists so it can be saved as JSON.
    """
    weight = fix_weight_sign(train_df["weight"])        # flip BEFORE the median
    coords = city_coordinates(train_df)
    return {
        "n_rows": int(len(train_df)),                   # evidence: how many rows it learned from
        "weight_median": float(weight.median()),        # D11 (median skips NaN)
        "city_coords": {city: [float(row.lat), float(row.lon)]
                        for city, row in coords.iterrows()},   # D15
    }


def fill_market_index(df: pd.DataFrame) -> pd.Series:
    """D12: fill missing market_index with the same-date mean from THIS file.

    Safety nets (never triggered in train or validation, per Step 7):
      - a date with no values at all  -> previous date's mean (ffill)
      - the very first date empty too -> next date's mean (bfill)
    Uses features only, never prices, so it is safe on any file.
    """
    date_mean = df.groupby(DATE_COL)["market_index"].mean().sort_index()  # mean skips NaN
    date_mean = date_mean.ffill().bfill()
    return df["market_index"].fillna(df[DATE_COL].map(date_mean))


def add_coordinates(df: pd.DataFrame, learned: dict) -> pd.DataFrame:
    """D15: if a file has city names but no coordinates (December), look them up
    from the training cities. Files that already have coordinates are left as-is."""
    if "pickup_lat" in df.columns:
        return df
    lookup = learned["city_coords"]
    unknown = (set(df["pickup"]) | set(df["delivery"])) - set(lookup)
    if unknown:
        raise ValueError(f"No learned coordinates for: {sorted(unknown)}")
    for end in ["pickup", "delivery"]:
        # .astype(float) guarantees numeric columns, whatever .map() infers
        df[f"{end}_lat"] = df[end].map(lambda c: lookup[c][0]).astype(float)
        df[f"{end}_lon"] = df[end].map(lambda c: lookup[c][1]).astype(float)
    return df


def daily_market_index(cleaned_df: pd.DataFrame) -> pd.Series:
    """D17: mean market_index per date, from an already-cleaned file.
    Used to give the December file the real December market level from validation."""
    return cleaned_df.groupby(DATE_COL)["market_index"].mean()


def clean(df: pd.DataFrame, learned: dict,
          market_by_date: pd.Series | None = None) -> pd.DataFrame:
    """Apply D10-D17 to a COPY of any file. Never drops or reorders rows.

    market_by_date is only needed for files WITHOUT a market_index column
    (the December file): pass daily_market_index(cleaned_validation).
    """
    out = df.copy()                                    # the caller's data stays untouched

    out[DATE_COL] = pd.to_datetime(out[DATE_COL], errors="raise")  # bad date -> stop

    unknown = set(out["equipment"].unique()) - KNOWN_EQUIPMENT       # D14
    if unknown:
        raise ValueError(f"Unknown equipment values: {sorted(unknown)}")

    out["weight"] = fix_weight_sign(out["weight"])                   # D10
    out["weight"] = out["weight"].fillna(learned["weight_median"])   # D11

    if "market_index" in out.columns:                                # D12
        out["market_index"] = fill_market_index(out)
    else:                                                            # D17 (December)
        if market_by_date is None:
            raise ValueError("No market_index column: pass market_by_date")
        out["market_index"] = out[DATE_COL].map(market_by_date)
        if out["market_index"].isna().any():
            raise ValueError("market_by_date has no value for some dates")

    out = add_coordinates(out, learned)                              # D15 (December)
    return out