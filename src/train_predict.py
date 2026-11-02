"""
train_predict.py: ONE command that produces both deliverables (D1, D56).

    python src/train_predict.py          (run from the project root)

The model is fixed by D55; this script makes no modelling choices.
  1. Final fit on ALL labeled rows (Jan-Oct): fit_cleaning -> clean ->
     drop label errors (D13, D53) -> fit. Always refit, nothing saved (D57).
  2. Predict the 12,000 validation loads -> validation_predictions.csv,
     placed by load_id in the template's row order, in cents (D58).
  3. Predict the 31 December rows -> outputs/december_chart_inputs.csv:
     original text copied, only predicted_rate filled (D7, D59).
  4. Re-read both WRITTEN files and check them; save every number to
     outputs/phase6_final_run.csv (D62).
Any failed check stops the run with an error.
"""
from __future__ import annotations

import hashlib
from functools import partial

import numpy as np
import pandas as pd

import baselines
import features
import models
from config import (DATE_COL, DECEMBER_OUTPUT_PATH, DECEMBER_PATH, ID_COL,
                    PHASE6_RUN_PATH, PREDICTIONS_PATH, TARGET, TEMPLATE_PATH,
                    TRAIN_PATH, VALID_PATH)

# D55: the submitted model.
MODEL_FACTORY = partial(models.HGBRate, target="log_rpm", extra=("weekday",), drop=("market_index",))
EXPECTED_NAME = "HGB log_rpm -market_index +weekday"
EXPECTED_COLUMNS = ["distance", "weight", "pickup_lat", "pickup_lon", "delivery_lat",
                    "delivery_lon", "eq_Dry Van", "eq_Flatbed", "eq_Reefer", "weekday"]
DEC_COLUMNS = ["pickup", "delivery", "distance", "equipment", "weight", "date", "predicted_rate"]
DEC_INPUTS = DEC_COLUMNS[:-1]
EXPECTED_ROWS = 12_000
SANITY_BAND = (750.0, 950.0)                     # Phase 2 expectation; reported, never used to change the model
CSV = {"index": False, "lineterminator": "\n"}   # D60: no index column, same bytes on every OS
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

summary: list[dict] = []                         # every reported number lands here (D62)


def check(name: str, ok: bool, detail: str = "") -> None:
    """Print PASS/FAIL; stop on the first failure."""
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        raise SystemExit(f"Check failed: {name}")


def record(section: str, item: str, value) -> None:
    summary.append({"section": section, "item": item, "value": value})


def sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def spread(values) -> dict:
    v = pd.Series(np.asarray(values, dtype=float))
    return {"median": v.median(), "p10": v.quantile(0.10), "p90": v.quantile(0.90)}


def final_fit(**settings):
    """Step 1. `settings` exists only for the Batch B seed test (random_state=1)."""
    raw = pd.read_csv(TRAIN_PATH)
    learned = features.fit_cleaning(raw)                  # training rows only (D18)
    train = features.clean(raw, learned)
    kept, flags = baselines.drop_label_errors(train)      # table learned from these rows (D13)
    model = MODEL_FACTORY(**settings).fit(kept)
    return raw, learned, train, kept, flags, model


def predict_validation(model, learned):
    valid = features.clean(pd.read_csv(VALID_PATH), learned)
    return valid, np.asarray(model.predict(valid), dtype=float)


def main() -> None:
    # ---------------- Step 1: final fit ----------------
    print("STEP 1  Final fit on all labeled rows")
    raw, learned, train, kept, flags, model = final_fit()
    dates = pd.to_datetime(raw[DATE_COL])
    n_drop, n_low, n_high = (int(flags[c].sum()) for c in ["is_error", "low", "high"])
    check("fit_cleaning learned from every training row", learned["n_rows"] == len(raw),
          f"{learned['n_rows']:,} rows, {dates.min().date()} to {dates.max().date()}")
    check("fit rows = all rows - label errors", len(kept) == len(raw) - n_drop,
          f"dropped {n_drop:,} (low {n_low}, high {n_high}); fit rows {len(kept):,}")
    check("model name matches D55", model.name == EXPECTED_NAME, model.name)
    check("feature columns and one-hot order fixed", model.columns_ == EXPECTED_COLUMNS,
          ", ".join(model.columns_))
    for item, value in [("rows_learned", learned["n_rows"]), ("first_date", str(dates.min().date())),
                        ("last_date", str(dates.max().date())), ("label_errors_dropped", n_drop),
                        ("dropped_low", n_low), ("dropped_high", n_high), ("fit_rows", len(kept)),
                        ("model_name", model.name), ("feature_columns", "|".join(model.columns_))]:
        record("fit", item, value)

    # ---------------- Step 2: validation predictions ----------------
    print("\nSTEP 2  Validation predictions")
    valid, pred = predict_validation(model, learned)
    check("one finite prediction > 0 per validation row",
          len(pred) == len(valid) and bool(np.isfinite(pred).all()) and bool((pred > 0).all()),
          f"{len(pred):,} predictions")
    template = pd.read_csv(TEMPLATE_PATH)
    check("template columns are load_id,predicted_rate", list(template.columns) == [ID_COL, "predicted_rate"])
    check("IDs unique in validation and template", valid[ID_COL].is_unique and template[ID_COL].is_unique)
    check("template ID set equals validation ID set", set(template[ID_COL]) == set(valid[ID_COL]),
          f"{len(template):,} IDs")

    exact_by_id = pd.Series(pred, index=valid[ID_COL].to_numpy())
    out = template.copy()
    out["predicted_rate"] = out[ID_COL].map(exact_by_id).round(2)   # by load_id; template order kept
    out.to_csv(PREDICTIONS_PATH, float_format="%.2f", **CSV)

    back = pd.read_csv(PREDICTIONS_PATH)                             # check what is ON DISK
    v = back["predicted_rate"].to_numpy(dtype=float)
    diff = np.abs(v - back[ID_COL].map(exact_by_id).to_numpy(dtype=float)).max()
    check("file columns exactly load_id,predicted_rate (no index)", list(back.columns) == [ID_COL, "predicted_rate"])
    check("file has 12,000 rows", len(back) == EXPECTED_ROWS, f"{len(back):,}")
    check("file keeps the template's row order", back[ID_COL].tolist() == template[ID_COL].tolist())
    check("no duplicate IDs in file", back[ID_COL].is_unique)
    check("written values finite and > 0", bool(np.isfinite(v).all()) and bool((v > 0).all()),
          f"min ${v.min():,.2f}, max ${v.max():,.2f}")
    check("written values = predictions to the cent", diff <= 0.005 + 1e-9, f"max diff {diff:.4f}")
    record("validation", "rows", len(back))
    record("validation", "pred_min", round(float(v.min()), 2))
    record("validation", "pred_max", round(float(v.max()), 2))

    # ---------------- Step 3: December file ----------------
    print("\nSTEP 3  December chart file")
    dec_text = pd.read_csv(DECEMBER_PATH, dtype=str, keep_default_na=False)   # exact original text
    check("original December columns in original order", list(dec_text.columns) == DEC_COLUMNS)
    dec = features.clean(pd.read_csv(DECEMBER_PATH), learned,
                         market_by_date=features.daily_market_index(valid))   # D17; unused by D55
    lined_up = (dec[DATE_COL].to_numpy() == pd.to_datetime(dec_text[DATE_COL]).to_numpy()).all()
    check("cleaned rows line up with original text rows", bool(lined_up))
    dec_pred = np.asarray(model.predict(dec), dtype=float)
    check("31 finite December predictions > 0",
          len(dec_pred) == 31 and bool(np.isfinite(dec_pred).all()) and bool((dec_pred > 0).all()))
    dec_out = dec_text.copy()
    dec_out["predicted_rate"] = [f"{x:.2f}" for x in dec_pred]                 # only this column changes
    dec_out.to_csv(DECEMBER_OUTPUT_PATH, **CSV)

    dback = pd.read_csv(DECEMBER_OUTPUT_PATH, dtype=str, keep_default_na=False)
    d = pd.to_datetime(dback[DATE_COL])
    r = pd.to_numeric(dback["predicted_rate"]).to_numpy(dtype=float)
    check("written file keeps the 7 columns in order", list(dback.columns) == DEC_COLUMNS)
    check("31 rows", len(dback) == 31)
    check("dates 2025-12-01..31, no duplicates",
          d.is_unique and set(d) == set(pd.date_range("2025-12-01", "2025-12-31")))
    check("6 input columns identical to data/ original (as text)", dback[DEC_INPUTS].equals(dec_text[DEC_INPUTS]))
    check("predicted_rate finite and > 0", bool(np.isfinite(r).all()) and bool((r > 0).all()))
    in_band = bool(((r >= SANITY_BAND[0]) & (r <= SANITY_BAND[1])).all())
    print(f"  {'PASS' if in_band else 'FLAG'}  inside sanity band ${SANITY_BAND[0]:.0f}-${SANITY_BAND[1]:.0f}"
          f"  (min ${r.min():.2f} on {d[r.argmin()].date()}, max ${r.max():.2f} on {d[r.argmax()].date()})")

    wk = pd.DataFrame({"date": d.dt.date, "day": d.dt.day_name(), "rate": r})
    print("\n  December, day by day:")
    print(wk.to_string(index=False))
    by_day = wk.groupby("day")["rate"].mean().reindex(WEEKDAYS).round(2)
    print("\n  Mean by weekday:")
    print(by_day.to_string())
    print(f"  highest: {by_day.idxmax()}, lowest: {by_day.idxmin()}")
    for item, value in [("min", round(float(r.min()), 2)), ("min_date", str(d[r.argmin()].date())),
                        ("max", round(float(r.max()), 2)), ("max_date", str(d[r.argmax()].date())),
                        ("mean", round(float(r.mean()), 2)), ("in_sanity_band", in_band),
                        ("highest_weekday", by_day.idxmax()), ("lowest_weekday", by_day.idxmin())]:
        record("december", item, value)
    for day, value in by_day.items():
        record("december_weekday_mean", day, value)

    # ---------------- Step 4: validation sanity (no labels -> shapes, not accuracy) ----------------
    print("\nSTEP 4  Validation sanity: predicted rates vs Jan-Oct training rates (report only)")
    rows = []
    for eq in ["all"] + models.EQUIPMENT_ORDER:
        t = train if eq == "all" else train[train["equipment"] == eq]
        mask = np.ones(len(valid), bool) if eq == "all" else (valid["equipment"] == eq).to_numpy()
        ts, ps = spread(t[TARGET]), spread(pred[mask])
        rows.append({"equipment": eq, "train_rows": len(t), "valid_rows": int(mask.sum()),
                     "train_p10": ts["p10"], "pred_p10": ps["p10"],
                     "train_median": ts["median"], "pred_median": ps["median"],
                     "train_p90": ts["p90"], "pred_p90": ps["p90"],
                     "train_med_miles": t["distance"].median(),
                     "valid_med_miles": valid.loc[mask, "distance"].median()})
    sanity = pd.DataFrame(rows).round(2)
    print(sanity.to_string(index=False))
    for row in sanity.to_dict("records"):
        for col, value in row.items():
            if col != "equipment":
                record(f"sanity_{row['equipment']}", col, value)

    month = valid[DATE_COL].dt.strftime("%Y-%m").to_numpy()
    nd = pd.DataFrame({"rows": pd.Series(pred).groupby(month).size(),
                       "pred_median": pd.Series(pred).groupby(month).median().round(2),
                       "med_miles": valid["distance"].groupby(month).median()})
    print("\n  November vs December:")
    print(nd.to_string())
    for m, row in nd.iterrows():
        for col, value in row.items():
            record(f"month_{m}", col, value)

    # ---------------- Files and summary ----------------
    record("files", "validation_predictions_sha256", sha256(PREDICTIONS_PATH))
    record("files", "december_sha256", sha256(DECEMBER_OUTPUT_PATH))
    pd.DataFrame(summary).to_csv(PHASE6_RUN_PATH, **CSV)
    print("\nWROTE")
    for p in [PREDICTIONS_PATH, DECEMBER_OUTPUT_PATH, PHASE6_RUN_PATH]:
        print(f"  {p.name:32s} sha256 {sha256(p)[:16]}...  ({p.parent.name}/)")
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()