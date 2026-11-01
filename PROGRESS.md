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
- Day 1 total: 0.75 + 1.9 + 1.4 = 4.05h
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
- Hours: Phase 4 = 1.5h
- Running total: 4.05 + 1.5 = 5.55 / 8.0