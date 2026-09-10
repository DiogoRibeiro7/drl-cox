# DRL-Cox

[![CI](https://github.com/DiogoRibeiro7/drl-cox/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/DiogoRibeiro7/drl-cox/actions/workflows/ci.yml)
[![Docs](https://github.com/DiogoRibeiro7/drl-cox/actions/workflows/docs.yml/badge.svg)](https://diogoribeiro7.github.io/drl-cox/)
[![codecov](https://codecov.io/gh/DiogoRibeiro7/drl-cox/branch/main/graph/badge.svg)](https://codecov.io/gh/DiogoRibeiro7/drl-cox)
[![Python 3.11 | 3.12 | 3.13](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue.svg)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Checked with mypy](https://img.shields.io/badge/mypy-checked-blue)](https://mypy-lang.org/)

Distributionally robust Cox regression with Wasserstein ambiguity sets, in Python.

`drl-cox` implements the **DRL-Cox** model introduced by Yeping Jin, Lauren Wise and
Ioannis Ch. Paschalidis in *Distributionally Robust Learning in Survival Analysis*
(CHIL 2025, PMLR 287; [arXiv:2506.01348](https://arxiv.org/abs/2506.01348)).
The Cox partial likelihood is replaced by a worst-case objective over a Wasserstein
ball around the empirical distribution, which yields a tractable exponential-cone
program that is more robust to outliers, covariate shift and model misspecification.

This is an independent implementation by Diogo Ribeiro. The authors' original code
lives at [noc-lab/drl_cox](https://github.com/noc-lab/drl_cox), and their CC BY 4.0
preprint is kept in [`paper/`](paper/) for convenience.

## Features

- **DRL-Cox solver**: exponential-cone program built with [CVXPY](https://www.cvxpy.org/).
  Uses [Clarabel](https://clarabel.org/) by default; SCS, ECOS and MOSEK are supported.
- **Baselines**: Cox partial likelihood (Newton-Raphson), ridge- and lasso-penalised Cox.
- **Metrics for censored data**: Harrell's C-index and IPCW time-dependent AUC / iAUC.
- **Selecting the Wasserstein radius**: grid cross-validation (sequential or parallel
  with joblib) and Bayesian optimisation with scikit-optimize.
- **Data utilities**: synthetic Cox data, a WHAS500-style CSV loader, contamination
  injectors (covariate shift, outliers) and survival-aware preprocessing (scaling,
  imputation, outlier detection, stratified splits).
- **scikit-learn integration**: `DRLCoxEstimator` works in pipelines and grid searches.
- **Plotting**: survival curves by risk group, coefficient paths, CV summaries.

## Installation

Requires Python 3.11, 3.12 or 3.13.

```bash
pip install drl-cox
```

For the legacy ECOS solver (Python 3.11 and 3.12 only):

```bash
pip install "drl-cox[ecos]"
```

From source, with [Poetry](https://python-poetry.org/) 2:

```bash
git clone https://github.com/DiogoRibeiro7/drl-cox.git
cd drl-cox
poetry install
```

## Quick start

```python
from drl_cox import concordance_index, fit_drl_cox, simulate_cox_data

data = simulate_cox_data(n=200, d=10, seed=42)

result = fit_drl_cox(
    data,
    epsilon=0.1,  # Wasserstein radius (0 recovers the unregularised problem)
    p=2.0,  # norm of the ground metric
    gamma=3,  # risk-set constraints per observation
)

risk = data.X @ result.beta
print(result.status, concordance_index(risk, data.y, data.zeta))
```

### Choosing `epsilon`

```python
from drl_cox import cross_validate_epsilon

cv = cross_validate_epsilon(data, epsilons=[0.0, 0.05, 0.1, 0.2], kfolds=5)
best = cv.groupby("epsilon")["score"].mean().idxmax()
```

`drl_cox.parallel_cv.cross_validate_epsilon` runs the same grid across CPU cores, and
`drl_cox.auto_select_epsilon` replaces the grid with Bayesian optimisation.

### scikit-learn API

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from drl_cox import DRLCoxEstimator

model = Pipeline([("scale", StandardScaler()), ("cox", DRLCoxEstimator(epsilon=0.1))])
model.fit(data.X, data.y, cox__zeta=data.zeta)
risk = model.predict(data.X)
```

### Data format

- `X`: numeric covariates, shape `(n_samples, n_features)`.
- `y`: positive survival or censoring times, shape `(n_samples,)`.
- `zeta`: event indicator, `1` if the event was observed and `0` if censored.

`SurvivalDataset(X, y, zeta)` validates these on construction; pass `validate=False`
to wrap raw data that still needs preprocessing.

## Documentation

Full documentation is published at <https://diogoribeiro7.github.io/drl-cox/>:

- [Installation](docs/installation.md)
- [Quick start](docs/quickstart.md)
- [Examples](docs/examples.md) and runnable scripts in [`examples/`](examples/)
- [scikit-learn API](docs/sklearn_api.md)
- [Parallel cross-validation](docs/parallel_cv_optimization.md)
- [API overview](docs/api.md)

## Development

```bash
poetry install --with docs
poetry run pre-commit install

poetry run pytest                # tests with coverage
poetry run ruff check . && poetry run ruff format --check .
poetry run mypy                  # strict type checking of src/
poetry run mkdocs serve          # documentation preview
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the workflow, and [CHANGELOG.md](CHANGELOG.md)
for release notes. Releases are published to PyPI automatically when a `vX.Y.Z` tag is
pushed.

## Citing

If you use this package, please cite both the software and the paper it implements.
Citation metadata for the software is in [CITATION.cff](CITATION.cff).

```bibtex
@inproceedings{jin2025drlcox,
  title     = {Distributionally Robust Learning in Survival Analysis},
  author    = {Jin, Yeping and Wise, Lauren and Paschalidis, Ioannis Ch.},
  booktitle = {Proceedings of the Conference on Health, Inference, and Learning (CHIL)},
  series    = {Proceedings of Machine Learning Research},
  volume    = {287},
  pages     = {1--12},
  year      = {2025},
  url       = {https://proceedings.mlr.press/v287/jin25a.html}
}
```

## License

MIT. See [LICENSE](LICENSE).
