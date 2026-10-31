# Progress

## Day 1 (2026-10-31)
### Phase 1 Setup: DONE
- Originals backed up outside project; files renamed to README names in data/
- Git initialised; .gitignore covers data/*.csv, .venv/, scratch/, __pycache__/, .ipynb_checkpoints/, notes/, Spotter PDF + README
- venv: Python 3.11.9; sklearn 1.9.1, pandas 2.3.3, numpy 2.4.6, matplotlib 3.11.2
- check_data.py: 0 mismatches (48,000 x 14 train, Jan 1 to Oct 31; 12,000 x 13 validation, Nov 1 to Dec 31; template IDs = validation IDs; 31-row December file; no ID overlap)
- score.py dummy run: valid files pass and chart renders; broken file (11,999 rows) rejected
- Quiz bank started: notes/QUIZ_BANK.md has 3 Phase 1 Q&As (git-ignored). Quizzes deferred: 5–8 tagged Q&As per phase; full compiled quiz + top 10 at the end
- Notes for later: `date` loads as text (must parse); weight is float (missing values); score.py chart y-axis zooms to the prediction range
- Hours: Phase 1 = 0.75h
- Running total: 0.75 / 8.0