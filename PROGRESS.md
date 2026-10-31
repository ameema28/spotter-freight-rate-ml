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
- Decisions D10–D18 logged
- src/features.py: fit_cleaning() + clean() (+ add_coordinates, daily_market_index); tested on train, validation and December; fold rehearsal (Jan–Jul → Aug–Oct) confirms no leakage; 0 gaps in model columns
- Bug found and fixed: test row-selection returned object dtype; coordinates also forced to float64 in add_coordinates
- Day-3 sanity band for December: ≈750–950, small weekly wiggle (chart y-axis zooms)
- Quiz: 8 Phase 2 Q&As appended to notes/QUIZ_BANK.md
- Hours: Phase 2 = __h
- Running total: __ / 8.0

