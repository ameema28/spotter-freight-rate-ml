# Decisions

| # | Phase | Decision | Choice | Reason |
|---|---|---|---|---|
| D1 | Setup | Project layout | Notebooks for EDA; `src/` .py scripts for final pipeline; shared `features.py` | Reviewer can rerun with one command; same cleaning in train and predict |
| D2 | Setup | Environment | venv, Python 3.11.9, Windows 11 | Built-in, lightweight; only pip needed by reviewer |
| D3 | Setup | File names | Renamed to README names in `data/` (underscores) | Spotter's documented score.py command works as written |
| D4 | Setup | Data in repo | Inputs, PDF and Spotter README git-ignored; outputs committed | Data belongs to Spotter; reviewers already have it; originals backed up locally |
| D5 | Setup | Requirements | Spotter's ranges kept; `scikit-learn==1.9.1` pinned exactly | The library that produces predictions must match exactly for reproducible results |
| D6 | Setup | Random seed | `SEED = 42` in `config.py`, passed as `random_state` everywhere (+ `np.random.seed`) | Identical results every run; later rerun with 2–3 seeds to show stability |