# drl-cox — Distributionally Robust Cox Regression (Wasserstein)

A complete, runnable Python repo that includes:

1. **DRL‑Cox solver** (cvxpy, exponential‑cone program)
2. **Standard Cox baselines** (partial likelihood; Ridge/Lasso via coordinate descent)
3. **Data utilities** to load WHAS500-like CSVs and **contamination helpers** (distributional shift, outliers)
4. **Poetry** project config + **GitHub Actions CI**
5. **Metrics** (C-index, time-dependent iAUC/IPCW)

> Code is written in English; docstrings, types, runtime checks, and inline comments included.

---

## File tree

```
drl-cox/
├─ pyproject.toml
├─ README.md
├─ src/
│  └─ drl_cox/
│     ├─ __init__.py
│     ├─ drl_cox.py            # DRL-Cox solver & CV utilities (from paper)
│     ├─ estimator.py          # High-level DRLCoxEstimator
│     ├─ cox_baseline.py       # Classical Cox models (PL, Ridge/Lasso)
│     ├─ metrics.py            # C-index, iAUC (IPCW)
│     ├─ datasets.py           # WHAS500 loader + synthetic generator
│     └─ contamination.py      # Shift & outlier injection helpers
├─ examples/
│  ├─ demo_whas500.py
│  └─ demo_estimator.py
└─ .github/
   └─ workflows/
      └─ ci.yml
```

## Overview

Distributionally Robust Cox Regression (Wasserstein) with:

- DRL-Cox convex program (Eq. (5) in CHIL 2025 paper)
- Classical Cox baselines (partial likelihood; Ridge/Lasso using coordinate descent)
- Metrics: C-index, iAUC (IPCW)
- Data helpers (WHAS500-like CSV loader) and contamination utilities (shift/outliers)

## Install

```bash
poetry install
poetry run python -m pip install -U pip
````

## Quick start

```bash
poetry run python examples/demo_whas500.py
```

The demo:

* Loads a CSV with columns `y`, `zeta` and covariates `x1..xd` (or uses synthetic data if not found)
* Fits DRL-Cox (ε tuned via CV)
* Fits standard Cox baselines
* Prints C-index and iAUC

## Data format (CSV)

* `y`: positive durations
* `zeta`: 1 if event observed, 0 if censored
* Covariates: numeric columns (e.g., `x1`, `x2`, ...)

## Notes

* DRL-Cox requires an exponential-cone capable solver (ECOS or SCS).
* Use `gamma` to control constraint windowing (O(gamma·N) constraints).
