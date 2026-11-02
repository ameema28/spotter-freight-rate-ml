"""
report_numbers.py: print EVERY number the report uses, read from the saved output
files only (no data files, no retraining). Lets anyone trace a report number to a row.

    python src/report_numbers.py          (run from the project root)

Each value is compared with the value written in the report plan
(HANDOFF_P6_P7 "Report number -> source"): MATCH / MISMATCH.
Numbers that exist only in DECISIONS.md are printed as constants, labelled
"DECISIONS.md (not recomputed)".
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

OUT = Path(__file__).resolve().parent.parent / "outputs"   # works from any folder

B1 = "B1 median rpm by equipment x band"
D55 = "HGB log_rpm -market_index +weekday"                 # the submitted model
TUNED = "HGB log_rpm -market_index +weekday learning_rate=0.05 max_iter=400"   # Phase 4, superseded
MARKET = "HGB log_rpm +market_28d learning_rate=0.05 max_iter=400"             # dropped (D40)
WALK = ["wf-May", "wf-Jun", "wf-Jul", "wf-Aug"]

mismatches: list[str] = []


def show(label: str, value, expected=None, ndp: int = 2) -> None:
    """Print one number; if the plan has a value for it, compare after rounding."""
    text = f"{value:,.{ndp}f}" if isinstance(value, float) else f"{value:,}" if isinstance(value, int) else str(value)
    tag = ""
    if expected is not None:
        same = round(float(value), ndp) == round(float(expected), ndp) if isinstance(value, (int, float)) else value == expected
        tag = "MATCH" if same else f"MISMATCH (plan says {expected})"
        if not same:
            mismatches.append(label)
    print(f"  {label:<58s} {text:>14s}   {tag}")


def section(title: str) -> None:
    print(f"\n=== {title} ===")


def one(df: pd.DataFrame, **where) -> pd.Series:
    """Exactly one row matching every column=value pair, or stop."""
    mask = pd.Series(True, index=df.index)
    for col, val in where.items():
        mask &= df[col] == val
    rows = df[mask]
    if len(rows) != 1:
        raise SystemExit(f"Expected 1 row for {where}, found {len(rows)}")
    return rows.iloc[0]


p3 = pd.read_csv(OUT / "phase3_baseline_scores.csv")
p4 = pd.read_csv(OUT / "phase4_experiments.csv")
p5 = pd.read_csv(OUT / "phase5_stability.csv")
boot = pd.read_csv(OUT / "phase5_bootstrap.csv")
run = pd.read_csv(OUT / "phase6_final_run.csv", dtype=str)
dec = pd.read_csv(OUT / "december_chart_inputs.csv", parse_dates=["date"])


def logged(section_name: str, item: str) -> str:
    return one(run, section=section_name, item=item)["value"]


# ---------------------------------------------------------------- data & quality
section("DATA & QUALITY  (DECISIONS.md / PROGRESS.md, not recomputed)")
for label, value in [("train rows (Jan 1 - Oct 31)", 48_000), ("validation rows (Nov 1 - Dec 31)", 12_000),
                     ("December rows", 31), ("negative weights train / valid", "292 / 145"),
                     ("missing weight train / valid", "300 / 165"),
                     ("missing market_index train / valid", "374 / 249")]:
    show(label, value)
n_err, n_rows = int(logged("fit", "label_errors_dropped")), int(logged("fit", "rows_learned"))
section("LABEL ERRORS  (phase6_final_run.csv)")
show("label errors flagged in full training file", n_err, 677)
show("  priced < half / > twice the expected rate", f"{logged('fit', 'dropped_low')} / {logged('fit', 'dropped_high')}", "337 / 340")
show("  share of training rows (%)", 100 * n_err / n_rows, 1.41)

# ---------------------------------------------------------------- split
section("SPLIT DESIGN  (DECISIONS.md D20/D21/D37 + row counts from CSVs)")
show("Sep-Oct holdout train / test rows (D20)", "38,477 / 9,523")
show("  test rows in phase5 'main' row", int(one(p5, fold="main", model=D55)["rows"]), 9523, 0)
show("decision fold Jan-Jul -> Aug rows (D21)", "33,718 / 4,759")
for f in WALK:
    show(f"  {f} test rows", int(one(p5, fold=f, model=D55, drop_errors=True)["rows"]), ndp=0)

# ---------------------------------------------------------------- B1 vs submitted
section("WALK-FORWARD: B1 vs SUBMITTED (D55)   [B1 = phase4, D55 = phase5]")
print(f"  {'fold':<8}{'B1 MAE':>10}{'D55 MAE':>10}{'gap B1-D55':>12}{'B1 MAPE':>9}{'D55 MAPE':>10}  winner (MAE)")
b1_mae, d_mae, b1_mape, d_mape, wins = [], [], [], [], 0
for f in WALK:
    b = one(p4, fold=f, model=B1, drop_errors=True)
    d = one(p5, fold=f, model=D55, drop_errors=True)
    gap = b["MAE"] - d["MAE"]
    wins += gap > 0
    b1_mae.append(b["MAE"]); d_mae.append(d["MAE"]); b1_mape.append(b["MAPE_%"]); d_mape.append(d["MAPE_%"])
    print(f"  {f:<8}{b['MAE']:>10.2f}{d['MAE']:>10.2f}{gap:>+12.2f}{b['MAPE_%']:>8.2f}%{d['MAPE_%']:>9.2f}%  "
          f"{'D55' if gap > 0 else 'B1 (by $%.2f)' % -gap}")
mean = lambda xs: sum(xs) / len(xs)
show("B1 walk-forward mean MAE", mean(b1_mae), 165.67)
show("D55 walk-forward mean MAE", mean(d_mae), 144.44)
for f, v, e in zip(WALK, d_mae, [155.73, 212.18, 119.79, 90.07]):
    show(f"  D55 {f} MAE", v, e)
show("mean MAE gap (B1 - D55, + = D55 better)", mean(b1_mae) - mean(d_mae))
show("B1 / D55 walk-forward mean MAPE (%)", f"{mean(b1_mape):.2f} / {mean(d_mape):.2f}")
show("folds where D55 has lower MAE", f"{wins}/4")
show("D55 BEATS B1 ON EVERY FOLD (MAE)", "yes" if wins == 4 else "no")
show("D55 lower MAPE on every fold", "yes" if all(d < b for d, b in zip(d_mape, b1_mape)) else "no")

# ---------------------------------------------------------------- Sep-Oct
section("SEP-OCT HOLDOUT (main)")
b_all = one(p3, fold="main", model=B1, train_label_errors="excluded", scope="all")
b_cln = one(p3, fold="main", model=B1, train_label_errors="excluded", scope="clean-only")
show("B1 MAE  (phase3)", b_all["MAE"], 140.53)
show("B1 MAE  (phase4 cross-check)", one(p4, fold="main", model=B1)["MAE"], 140.53)
show("B1 MAPE %", b_all["MAPE_%"], 6.40)
show("B1 clean MAE", b_cln["MAE"], 85.01)
d = one(p5, fold="main", model=D55, drop_errors=True)
show("D55 MAE  (third look, report only)", d["MAE"], 98.57)
show("D55 MAPE %", d["MAPE_%"], 4.34)
show("D55 clean MAE", d["clean_MAE"], 42.53)
show("D55 bias, all rows ($, pred - true)", d["bias"])
show("D55 clean bias %  (DECISIONS.md D55, not recomputed)", "-0.77%")
show("  settings note in log", d["settings"])
m = one(p4, fold="main", model=MARKET)
show("market model MAE (D40, dropped)", m["MAE"], 188.31)
show("market model bias %  (DECISIONS.md D40, not recomputed)", "-5.6%")

# ---------------------------------------------------------------- stability
section("STABILITY: paired day-block bootstrap (phase5_bootstrap.csv)")
print("  gap = ALTERNATIVE minus SUBMITTED MAE; + means the submitted model is better")
s = one(boot, scope="pooled 4 folds", unit="day", comparison="SUBMITTED MAE")
show("submitted pooled MAE", s["point"], 144.42)
show("  day interval", f"[{s['ci_low']:.2f}, {s['ci_high']:.2f}]  ({int(s['n_units'])} days, "
                       f"{int(s['repeats'])} repeats, seed {int(s['seed'])})")
plan = {"ERRORS KEPT minus SUBMITTED": (7.59, 6.34, 8.68), "NO-WEEKDAY minus SUBMITTED": (1.42, -1.26, 3.91),
        "TUNED lr0.05x400 minus SUBMITTED": (-0.11, -0.34, 0.11)}
for comp, (pt, lo, hi) in plan.items():
    r = one(boot, scope="pooled 4 folds", unit="day", comparison=comp)
    alt = comp.split(" minus")[0]
    favours = "SUBMITTED better" if r["point"] > 0 else f"{alt} better"
    print(f"  -- {comp}")
    show("     gap", r["point"], pt)
    show("     95% interval low", r["ci_low"], lo)
    show("     95% interval high", r["ci_high"], hi)
    show("     direction", f"{favours} by ${abs(r['point']):.2f}")
    show("     folds same direction / label", f"{int(r['folds_same_direction'])}/4 / {r['label']}")
w = one(boot, scope="pooled 4 folds", unit="row", comparison="NO-WEEKDAY minus SUBMITTED")
show("weekday ROW-level interval (comparison only)", f"[{w['ci_low']:.2f}, {w['ci_high']:.2f}]")

# ---------------------------------------------------------------- tuned-model checks (labelled)
section("MEASURED ON THE PHASE 4 TUNED MODEL, NOT D55 (label in report)")
rand = [one(p4, fold=f"random s{s_}", model=TUNED)["clean_MAE"] for s_ in (42, 1, 2)]
show("D43 random split clean MAE min", min(rand), 67.49)
show("D43 random split clean MAE max", max(rand), 68.12)
show("D43 time split clean MAE (Sep-Oct)", one(p4, fold="main", model=TUNED)["clean_MAE"], 42.31)
w_in = one(p4, fold="wf-Aug held-out cities: WITH in training", model=TUNED)
w_out = one(p4, fold="wf-Aug held-out cities: WITHOUT in training", model=TUNED)
w_b1 = one(p4, fold="wf-Aug held-out cities: B1", model=B1)
show("D44 rows scored", int(w_in["rows"]), 1226, 0)
show("D44 MAE with cities in training", w_in["MAE"], 71.63)
show("D44 MAE without them", w_out["MAE"], 72.20)
show("D44 unseen-city cost (exact difference)", w_out["MAE"] - w_in["MAE"], 0.56)   # D73: D44's 0.57 came from rounded values
show("D44 B1 on same rows", w_b1["MAE"], 122.09)

# ---------------------------------------------------------------- final run
section("FINAL FIT & OUTPUTS (phase6_final_run.csv)")
show("rows learned / first / last date", f"{logged('fit', 'rows_learned')} / {logged('fit', 'first_date')} / {logged('fit', 'last_date')}")
show("fit rows", int(logged("fit", "fit_rows")), 47323, 0)
show("model name", logged("fit", "model_name"), D55)
show("validation rows", int(logged("validation", "rows")), 12000, 0)
show("validation pred min", float(logged("validation", "pred_min")), 184.61)
show("validation pred max", float(logged("validation", "pred_max")), 6821.37)
print(f"  {'equipment':<10}{'train':>7}{'valid':>7}   train p10 / median / p90        pred p10 / median / p90")
for eq in ["all", "Dry Van", "Flatbed", "Reefer"]:
    g = lambda c: float(logged(f"sanity_{eq}", c))
    print(f"  {eq:<10}{int(g('train_rows')):>7,}{int(g('valid_rows')):>7,}   "
          f"{g('train_p10'):,.2f} / {g('train_median'):,.2f} / {g('train_p90'):,.2f}   "
          f"{g('pred_p10'):,.2f} / {g('pred_median'):,.2f} / {g('pred_p90'):,.2f}")
show("pred median vs train median", f"{float(logged('sanity_all', 'pred_median')):,.2f} vs {float(logged('sanity_all', 'train_median')):,.2f}")
show("median miles valid vs train", f"{logged('sanity_all', 'valid_med_miles')} vs {logged('sanity_all', 'train_med_miles')}")
for mth in ["2025-11", "2025-12"]:
    show(f"{mth} rows / pred median / median miles",
         f"{int(float(logged('month_' + mth, 'rows'))):,} / {float(logged('month_' + mth, 'pred_median')):,.2f} / {logged('month_' + mth, 'med_miles')}")
show("sha256 validation_predictions.csv (prefix)", logged("files", "validation_predictions_sha256")[:16], "f7e54d41d43fb4a3")
show("sha256 december_chart_inputs.csv (prefix)", logged("files", "december_sha256")[:16], "7bf1bafca076f163")

# ---------------------------------------------------------------- December
section("DECEMBER (recomputed from outputs/december_chart_inputs.csv, checked vs log)")
r = dec["predicted_rate"]
show("rows", len(dec), 31, 0)
show("min (recomputed)", float(r.min()), float(logged("december", "min")))
show("  on", f"{dec.loc[r.idxmin(), 'date'].date()} ({dec.loc[r.idxmin(), 'date'].day_name()})")
show("max (recomputed)", float(r.max()), float(logged("december", "max")))
show("  on", f"{dec.loc[r.idxmax(), 'date'].date()} ({dec.loc[r.idxmax(), 'date'].day_name()})")
show("weekly range max - min", float(r.max() - r.min()), 19.40)
by_day = dec.groupby(dec["date"].dt.day_name())["predicted_rate"]
for day in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]:
    show(f"  {day} mean", float(by_day.mean()[day]), float(logged("december_weekday_mean", day)))
show("exact weekly repeat (each weekday one value)", "yes" if (by_day.nunique() == 1).all() else "no")
xmas = dec[dec["date"] == "2025-12-25"].iloc[0]
show("Christmas Day weekday / price", f"{xmas['date'].day_name()} / {xmas['predicted_rate']:.2f}")
show("Dec 1 / Dec 31", f"{r.iloc[0]:.2f} / {r.iloc[-1]:.2f}")
fixed = {c: sorted(dec[c].astype(str).unique()) for c in ["pickup", "delivery", "distance", "equipment", "weight"]}
show("each fixed input has ONE value", "yes" if all(len(v) == 1 for v in fixed.values()) else "no")
show("fixed inputs", " | ".join(v[0] for v in fixed.values()))

print(f"\nMISMATCHES: {len(mismatches)}" + (f"  -> {mismatches}" if mismatches else ""))
