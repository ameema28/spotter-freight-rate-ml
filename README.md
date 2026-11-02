# Freight rate prediction: Spotter ML Engineer assessment

Predicts the posted rate (in dollars) of a freight load. Trained on 48,000 labelled loads
(January to October 2025); predicts the 12,000 validation loads (November to December 2025)
and the 31-day December chart. Full write-up:
[report/Spotter_Freight_Rate_Report_Ameema_Rashid.pdf](report/Spotter_Freight_Rate_Report_Ameema_Rashid.pdf)

## Results

- **September–October checkpoint:** MAE **$98.57** vs **$140.53** for the baseline
  (median rate per mile by equipment and distance band); MAPE 4.34% vs 6.40%.
- **Walk-forward folds (test May, Jun, Jul, Aug, each trained on all earlier months):**
  lower MAE in 3 of 4 months (June a tie, +$0.04), mean $144.44 vs $165.67;
  lower MAPE in every month.
- The checkpoint changed one decision: dropping `market_index` (D40). The replacement
  was decided in advance, and every look at the checkpoint is disclosed in the report.
- Spotter's metric is not stated, so MAE in dollars is the primary metric; MAPE, RMSE and
  bias are also reported.

## The model

- scikit-learn `HistGradientBoostingRegressor`, library defaults, `early_stopping=False`,
  `random_state=42`.
- Target: log(rate per mile); dollars = exp(prediction) × distance.
- Features: distance, weight, pickup and delivery coordinates, equipment (one-hot), weekday.
- Cleaning, learned from training rows only: negative weights sign-flipped; missing weights
  filled with the training median; 677 rows priced 2–6× off similar loads (likely label
  errors) excluded from training only. Every test row is still scored.
- `market_index` and `quote_signal` have no documented meaning. Both were tested and
  dropped (D40, D34).

## Setup and run

Tested on Python 3.11.9, Windows 11, with numpy 2.4.6, pandas 2.3.3, matplotlib 3.11.2 and
scikit-learn 1.9.1 (pinned).

### 1. Place the data

The data files belong to Spotter and are not in this repository. Copy Spotter's four
downloads into `data/` (the folder exists after cloning) with these names:

| Spotter's download | Name in `data/` |
|---|---|
| train-test.csv | train_test.csv |
| validation.csv | validation.csv |
| validation-predictions-template.csv | validation_predictions_template.csv |
| december-chart-inputs.csv | december_chart_inputs.csv |

### 2. Run (from the project root)

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # macOS/Linux
pip install -r requirements.txt
python src/train_predict.py
python score.py --predictions validation_predictions.csv --december-predictions outputs/december_chart_inputs.csv
```

Optional: `python src/report_numbers.py` prints every number in the report from the saved
files in `outputs/` and ends with `MISMATCHES: 0`.

`train_predict.py` retrains on all labelled data (takes seconds; no model file is saved),
writes both deliverables, re-reads them from disk, checks them, and ends with
`ALL CHECKS PASSED`.

**Deliberate difference from Spotter's README:** the filled December file is written to
`outputs/december_chart_inputs.csv`, not `data/`, so the input files are never overwritten.
The `score.py` command above points to it.

### 3. Expected output

The output files are byte-identical on every operating system. SHA-256 prefixes:

| File | SHA-256 starts with |
|---|---|
| validation_predictions.csv | f7e54d41d43fb4a3 |
| outputs/december_chart_inputs.csv | 7bf1bafca076f163 |

On Windows: `Get-FileHash validation_predictions.csv -Algorithm SHA256`

## Outputs in this repository

| File | What it is |
|---|---|
| validation_predictions.csv | 12,000 rows, `load_id,predicted_rate`, in dollars to the cent |
| outputs/december_chart_inputs.csv | Spotter's December file with `predicted_rate` filled |
| scorer_results/candidate_december.png | December chart produced by `score.py` |
| outputs/phase6_final_run.csv | Every number from the final run |
| outputs/phase3_baseline_scores.csv, phase4_experiments.csv, phase5_stability.csv, phase5_bootstrap.csv | Experiment logs behind the report |
| report/Spotter_Freight_Rate_Report_Ameema_Rashid.pdf | Report: split and validation approach, results, December chart |

## Repository layout

| File | Purpose |
|---|---|
| src/config.py | Paths and settings (seed, distance bands, label-error thresholds) |
| src/check_data.py | First checks on the raw files: shapes, date ranges, ID matches |
| src/features.py | `fit_cleaning()` learns from training rows; `clean()` applies the same rules to any file |
| src/baselines.py | Baselines B0 and B1, distance bands, label-error flags |
| src/validation.py | Time-based splits, walk-forward folds, metrics, the per-fold pipeline |
| src/models.py | Feature building and the model classes (`HGBRate`) |
| src/experiments.py | Runs and logs walk-forward experiments |
| src/train_predict.py | One command: final fit, both output files, checks |
| src/report_numbers.py | Re-prints every report number from `outputs/` with MATCH/MISMATCH |
| score.py | Spotter's scorer, unchanged |
| DECISIONS.md | Every decision with its reason |
| PROGRESS.md | Work log and hours |

## Notebooks

`notebooks/02_eda.ipynb` to `05_stability.ipynb` hold the exploration and experiments
(install with `pip install -r requirements-dev.txt`). Notebook outputs predate D64, a code
tidy that reproduced the submitted model's fold results exactly; they will be re-run.
Re-running notebooks 04 and 05 may write to the experiment logs in `outputs/`.

## AI assistance

Same statement as report section 9:

> I used Claude (Anthropic) as a mentor throughout: it explained options and trade-offs,
> drafted code and the wording of this report from my logged results. I chose every
> option, ran every step myself, checked every output, and can explain each part of the
> code. Every number in this report is either printed by src/report_numbers.py from the
> saved output files or taken from the decision log (DECISIONS.md) in the repository.