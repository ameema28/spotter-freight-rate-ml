"""
experiments.py: one row per experiment in a log CSV (default outputs/phase4_experiments.csv; Phase 5 passes outputs/phase5_stability.csv).
"""
from __future__ import annotations

import validation
import pandas as pd
import numpy as np
from pathlib import Path
from config import OUTPUT_DIR, TARGET
from validation import score

LOG_PATH = OUTPUT_DIR / "phase4_experiments.csv"
COLUMNS = ["fold", "model", "target", "features", "drop_errors", "settings",
           "rows", "MAE", "MAPE_%", "RMSE", "bias", "clean_MAE"]
KEY = ["fold", "model", "target", "features", "drop_errors", "settings"]


def experiment_row(result: dict, target: str, features: str, settings: str) -> dict:
    """Turn a run_split/run_fold result into one log row (all rows + clean-only MAE)."""
    h = result["holdout"]
    clean = h[~h["is_label_error"]]
    return {"fold": result["fold"], "model": result["model"], "target": target,
            "features": features, "drop_errors": result["drop_errors"], "settings": settings,
            "rows": result["test_rows"], **result["scores"],
            "clean_MAE": score(clean[TARGET], clean["pred"])["MAE"]}


def log_experiments(rows: list[dict], path=None) -> pd.DataFrame:
    """Merge new rows into the CSV at `path`, replacing any row with the same KEY.
    path=None means LOG_PATH (the Phase 4 log), so Phase 4 code behaves exactly as before.
    (None, not LOG_PATH, as the default: a default value is frozen when Python reads the
    function, so None keeps it pointing at whatever LOG_PATH is when the call happens.)"""
    path = LOG_PATH if path is None else Path(path)
    new = pd.DataFrame(rows, columns=COLUMNS)
    if path.exists():
        old = pd.read_csv(path)
        key = lambda d: d[KEY].astype(str).agg("|".join, axis=1)
        new = pd.concat([old[~key(old).isin(key(new))], new], ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    new.round(4).to_csv(path, index=False)
    return new


def run_walk(raw, folds, make, target, features, settings, drop_errors=True, log_path=None) -> list[dict]:
    """Run one model on every walk-forward fold and log one row per fold to log_path (default: Phase 4 log)."""
    results = [validation.run_fold(raw, f, make, drop_errors) for f in folds]
    log_experiments([experiment_row(r, target, features, settings) for r in results], path=log_path)
    return results


def walk_summary(runs: dict, ref: str) -> pd.DataFrame:
    """Per-fold MAE + means, and the decision-15b rule vs `ref`:
    mean MAE improves >= $1 AND wins >= 3 of 4 folds AND mean MAPE rises <= 0.10 pp."""
    rows = []
    for name, res in runs.items():
        row = {"run": name, **{f"MAE {r['fold']}": r["scores"]["MAE"] for r in res}}
        cleans = [r["holdout"][~r["holdout"]["is_label_error"]] for r in res]
        row["mean_MAE"] = np.mean([r["scores"]["MAE"] for r in res])
        row["mean_MAPE_%"] = np.mean([r["scores"]["MAPE_%"] for r in res])
        row["mean_clean_MAE"] = np.mean([score(c[TARGET], c["pred"])["MAE"] for c in cleans])
        row["mean_clean_bias_%"] = np.mean([100 * ((c["pred"] - c[TARGET]) / c[TARGET]).mean() for c in cleans])
        rows.append(row)
    t = pd.DataFrame(rows).set_index("run")
    fold_cols = [c for c in t.columns if c.startswith("MAE ")]
    t["dMAE_vs_ref"] = t["mean_MAE"] - t.loc[ref, "mean_MAE"]
    t["dMAPE_pp_vs_ref"] = t["mean_MAPE_%"] - t.loc[ref, "mean_MAPE_%"]
    t["wins_vs_ref"] = (t[fold_cols] < t.loc[ref, fold_cols]).sum(axis=1)
    t["passes_rule"] = (t["dMAE_vs_ref"] <= -1) & (t["wins_vs_ref"] >= 3) & (t["dMAPE_pp_vs_ref"] <= 0.10)
    return t