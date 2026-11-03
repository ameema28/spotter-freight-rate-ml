# Progress

## Day 1 (2026-10-31)

### Phase 1 Setup: DONE
- Originals backed up outside project; files renamed to README names in data/
- Git initialised; .gitignore covers data/*.csv, .venv/, scratch/, __pycache__/, .ipynb_checkpoints/, notes/, Spotter PDF + README
- venv: Python 3.11.9; sklearn 1.9.1, pandas 2.3.3, numpy 2.4.6, matplotlib 3.11.2
- check_data.py: 0 mismatches (48,000 x 14 train, Jan 1 to Oct 31; 12,000 x 13 validation, Nov 1 to Dec 31; template IDs = validation IDs; 31-row December file; no ID overlap)
- score.py dummy run: valid files pass and chart renders; broken file (11,999 rows) rejected
- Quiz rule: Phase 1 kept at 3 Q&As by choice; 5–8 tagged Q&As per phase from Phase 2 on
- Spotter readme.md renamed SPOTTER_README.md (git-ignored); own README.md will be committed. GitHub repo not created yet (Day 3)
- Notes for later: `date` loads as text (must parse); weight is float (missing values); score.py chart y-axis zooms to the prediction range
- Hours: Phase 1 = 0.75h
- Running total: 0.75 / 8.0

### Phase 2 EDA & cleaning: DONE
- Checks 0–10 run in notebooks/02_eda.ipynb; every brief number reproduced or corrected
- Confirmed: 292/145 negative weights (sign errors); 300/165 missing weight; 374/249 missing market_index (random; validation ≈2–3× more gaps); no duplicates; dates continuous, validation ≈25% denser per day
- New findings: 70-mile distance floor; coordinates synthetic but consistent; 677 train rows (1.41%) priced 2–6× off (consistent with label errors); market_index = daily level + weekly cycle (Thu high, Sun/Mon low) + late-July drop; quote_signal flips by month and is noise in Aug (validation looks like Aug); 8 cities only in validation (12.1% rows), Laredo outside the training map
- Decisions D10–D19 logged
- src/features.py: fit_cleaning() + clean() (+ add_coordinates, daily_market_index); tested on train, validation and December; fold rehearsal (Jan–Jul → Aug–Oct) confirms no leakage; 0 gaps in model columns
- Bug found and fixed: test row-selection returned object dtype; coordinates also forced to float64 in add_coordinates; re-tested after fix (12c + Step 13), passed
- Day-3 sanity band for December: ≈750–950, small weekly wiggle (chart y-axis zooms)
- Distance bands recorded: [0, 250, 500, 1000, 2000, 4000] mi (max train distance 3,439.8); the 677 count is from the full file, not per fold
- Quiz: 8 Phase 2 Q&As appended to notes/QUIZ_BANK.md
- Hours: Phase 2 = 1.9h
- Running total: 2.65 / 8.0

### Phase 3 Split & baseline: DONE
- Step 0: DATE_COL imported from config; distance_band() (edges in config) with safety check; Phase 2 tests re-run, unchanged; notebook 02 broken test cell removed, Run All clean
- Step 1: time_split(); decision 33,718 → 4,759 (Aug); main 38,477 → 9,523 (Sep–Oct); overlap refused
- Step 2: MAE/MAPE/RMSE/bias, toy-checked by hand, match sklearn, bad inputs rejected
- Step 3: baselines.py; full-file flag reproduces 677 (337/340); per-fold flags 465 / 533 identical to full-file flags (robust rule)
- Step 4–6: run_fold/run_split; cleaning saw training rows only (33,718 / 38,477); B1 bar MAE $140.53, MAPE 6.40% (main, all rows)
- Findings: label errors = 1.5% rows, 96.7% of squared error; no Sep→Oct drift on clean rows; random split harsher for B1 (MAE $101 vs $85) due to monthly price levels (bias +$134 Jan, −$112 Jun) → market_index is the model's main job in Phase 4
- Quiz: 8 Phase 3 Q&As appended to notes/QUIZ_BANK.md
- Hours: Phase 3 = 1.4h
- Day 1 total = 4.05h
- Running total: 4.05 / 8.0

## Day 2 (2026-11-01)

### Phase 4 Features & first model: DONE
- Pre-checks: Nov–Dec market_index inside training range (Sep–Oct 0.25% below the Jan–Aug min, 26.8% in its low tail); quote_signal 0 missing; smallest price-table cell 341 / 386 / 481 rows
- src/models.py (build_features, S1, HGBRate), src/experiments.py (log, run_walk, walk_summary), features.trailing_market_mean, validation.WALK_FOLDS, config.MARKET_WINDOW_DAYS
- Checks passed: B1 reproduces 136.60 / 140.53; predict() blind to posted_rate (identical with the column deleted); trailing mean matches a hand calculation; gaps refused
- Main checkpoint 1 FAILED: market model 188.31 vs B1 140.53 (bias −5.6%) → market_index link broke after the late-July drop → dropped (D40)
- Main checkpoint 2 (pre-registered, disclosed): final model MAE 98.34 / MAPE 4.33% / clean 42.31 vs B1 140.53 / 6.40% / 85.01
- Random vs time: random harsher (67.5–68.1 vs 42.3 clean); unseen city cost $0.57; importance: distance > equipment > weight > locations > weekday
- December look: 811.84–831.44, weekday pattern (Thu high, Sun low); validation 12,000 preds finite > 0, median $2,031
- Bugs fixed: model name missing "-market_index" (fixed, re-checked); ratio print placeholder (read from $ values)
- Decisions D30–D46 logged; 63 experiment rows
- Quiz: skipped by PACE rule (built after submission)
- Hours: Phase 4 = 1.5h
- Running total: 5.55 / 8.0

### Phase 5 Stability checks: DONE
- experiments.log_experiments/run_walk take a log path; Phase 5 rows in outputs/phase5_stability.csv (27), bootstrap in outputs/phase5_bootstrap.csv (24); Phase 4 log unchanged (fingerprint)
- Determinism: seed 42 vs 1 on wf-Aug, max prediction difference 0 → seed repeats dropped (D48)
- Reproduced: final per-fold MAEs 155.81/212.15/119.38/89.99, Sep–Oct 98.34; NO-WEEKDAY matches Phase 4 on 4/4 folds
- Re-checks on the final model: tuned settings only −$0.11 → D39 REVERSED to library defaults (D52); label errors kept +$7.59 → D38 confirmed (D53); weekday +$1.42, 3/4 → kept (D54)
- Paired day-block bootstrap (1,000, seed 42): label errors robust [+6.34, +8.68]; weekday [−1.26, +3.91] and settings [−0.34, +0.11] within noise; row-level intervals far narrower (rows overstate certainty)
- Third look Sep–Oct (report only): submitted model 98.57 / 4.34% / clean 42.53 vs B1 140.53 / 6.40%
- Decisions D47–D55 logged; D6, D39, D42 amended
- Quiz: skipped by PACE rule (built after submission)
- Hours: Phase 5 = 0.62h
- Running total: 6.17 / 8.0

## Day 3 (2026-11-02)

### Phase 6 Final retrain & output files: DONE
- Pre-checks: pandas 2.3.3 to_csv lineterminator default None → '\r\n' on Windows; HGB defaults lr 0.1 / max_iter 100; after D64 edits D55 walk folds MATCH 155.73/212.18/119.79/90.07
- src/train_predict.py (one command): fit_cleaning 48,000 rows (2025-01-01..10-31) → 677 label errors dropped (337 low / 340 high) → 47,323 fit rows; name "HGB log_rpm -market_index +weekday"; 10 columns, one-hot order fixed
- validation_predictions.csv: 12,000 rows, template order, ID set = template, no duplicates, no index, all > 0 ($184.61–$6,821.37), cents (max diff 0.005); checked on the re-read file
- outputs/december_chart_inputs.csv: 6 input columns identical as text; 31 unique dates; $818.21 (Sun) – $837.61 (Thu); exact weekly repeat (only weekday varies); Christmas Day priced highest → holiday limitation
- Sanity (report only): pred median $2,029.69 vs train $2,030.76; median miles 953.6 vs 953.3; equipment medians close; Nov $2,044.24 vs Dec $2,019.15 (Dec loads ~13 mi shorter)
- Determinism: seed 42 vs 1 on the full fit, max diff 0.0 (validation + December); two runs byte-identical (SHA-256 f7e54d41…, 7bf1bafc…, fc48b52e…); no CR in files
- score.py passed (12,000 + 31 validated); chart checked against the December numbers; old dummy chart in scratch/ spotted and not used
- Requirements: all imports covered, pip check clean; .gitignore correct; PNG committed
- Bug fixed: f-string backslash (Python 3.11) in a scratch script; re-run passed
- Decisions D56–D64 logged; D48 amended
- Quiz: skipped by PACE rule (built after submission)
- Follow-up commit 482d2a5: market_rel path re-checked after the D64 finite-check move (runs on all 4 walk folds; raise check only); D64 amended
- Hours: Phase 6 = 0.33h
- Running total: 6.50 / 8.0

### Phase 7 Report (+ Loom script): DONE
- src/report_numbers.py: every report number printed from outputs/ CSVs with MATCH/MISMATCH vs the plan; "D55 beats B1 on every fold: no" (June B1 by $0.04) → wording changed (D72); D44 $0.57 → $0.56 (D73); final run MISMATCHES: 0
- Report: 4 A4 pages, DOCX → PDF, both required items (split/validation approach; score.py December chart from project-root PNG); tuned-model checks labelled; gap directions match phase5_bootstrap.csv signs
- External review fixes: summary overstated "every modelling choice" (D74); tuned-settings interval direction made explicit; rounding note under walk-forward table; "twomonth"/"reportonly" confirmed as text-extraction artefacts, not in the PDF
- Loom script notes/LOOM_SCRIPT.md: 5 required topics + AI disclosure
- .gitignore checked: report/ and src/report_numbers.py tracked; notes/ ignored
- Decisions D65–D74 logged; D44 amended
- Quiz: skipped by PACE rule (built after submission)
- Hours: Phase 7 = 0.52h
- Running total: 7.02 / 8.0

### Phase 8 Repo, README, fresh-clone test, Loom: DONE
- README.md + data/.gitkeep committed (1d0affe); README rendered; src/ lines match docstrings
- Pre-push history check: Spotter data + PDF found only on refs/agents/* checkpoint commits, not on main (D78); pushed main only; ls-remote: HEAD + main
- Repo public: github.com/ameema28/spotter-freight-rate-ml; logged-out view checked [ ✓ ]
- Fresh-clone test passed (D79): ALL CHECKS PASSED, hashes match, score.py passed, MISMATCHES: 0, committed bytes = regenerated bytes
- Loom recorded: [7:22], camera on, 5 topics + AI disclosure; link opens logged out ✓; Loom's auto-summary corrected (transcription errors: "$1,99", "fast loads")
- Report PDF re-exported (D81): Stability heading fixed, 4 pages checked
- Spotter's form: wrong project links submitted by accident (a different full-stack project); form closed → reopen requested and correct submission sent by email reply with attachments (D82)
- Tag submitted-v1 pushed after this commit
- Decisions D75–D82 logged
- Quiz: skipped by PACE rule (built after submission)
- Hours: Phase 8 = 0.5h
- FINAL total: 7.52 / 8.0