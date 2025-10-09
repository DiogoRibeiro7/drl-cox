# DRL-Cox: Distributionally Robust Survival Analysis

This repository implements the Wasserstein distributionally robust Cox regression model (DRL-Cox) together with classical baselines, evaluation metrics, and reproducible demo scripts.

## Features

- DRL-Cox solver built with `cvxpy` and exponential cone constraints.
- Baseline Cox proportional hazards estimators (partial likelihood, ridge, lasso).
- Metrics for censored survival data, including C-index and time-dependent iAUC/IPCW.
- Data loaders and contamination utilities for WHAS500-style datasets.
- Poetry project configuration and continuous integration workflow.

## Repository Layout

```
drl-cox/
|-- paper/                               # Publication and supplementary material
|   `-- Distributionally Robust Learning in Survival Analysis.pdf
|-- examples/                            # End-to-end usage demos
|-- src/drl_cox/                         # Library code
|   |-- cox_baseline.py
|   |-- contamination.py
|   |-- datasets.py
|   |-- drl_cox.py
|   |-- estimator.py
|   `-- __init__.py
|-- .github/workflows/ci.yml             # CI pipeline
|-- pyproject.toml
`-- README.md
```

## Installation

```bash
poetry install
```

The project targets Python 3.11+. Poetry will install the solver dependencies (ECOS and SCS) declared in `pyproject.toml`.

## Quick Start

Run the WHAS500 demo to fit DRL-Cox and baseline models:

```bash
poetry run python examples/demo_whas500.py
```

The script loads WHAS500-style data (or generates synthetic samples if the CSV is missing), trains DRL-Cox with cross-validated regularization, fits the classical Cox models, and prints the survival metrics.

## Data Format

- `y`: positive survival durations.
- `zeta`: event indicator (1 if observed, 0 if censored).
- Covariates: numerical columns such as `x1`, `x2`, ... .

## Paper

The accompanying manuscript, *Distributionally Robust Learning in Survival Analysis*, is stored under `paper/`. Keep supplementary figures and future revisions in the same directory for easy reference.

## Development Notes

- DRL-Cox requires an exponential-cone capable solver. ECOS is the default; uncomment SCS in `pyproject.toml` if needed.
- Adjust the `gamma` parameter to trade off robustness and computational cost; larger values add more constraints.
- Use the CI workflow (`.github/workflows/ci.yml`) as a reference when extending tests or adding lint checks.
