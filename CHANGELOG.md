# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- **Default solver is now Clarabel** (`DEFAULT_SOLVER = "CLARABEL"`). CVXPY no longer
  installs ECOS, so the previous default failed on a fresh install. ECOS is available
  through the `drl-cox[ecos]` extra on Python 3.11 and 3.12. Solver options are passed
  through unchanged; Clarabel uses `max_iter` where ECOS and SCS use `max_iters`.
- Supported Python versions are 3.11, 3.12 and 3.13. Python 3.10 was never installable
  with the declared dependencies.
- Project metadata moved to the PEP 621 `[project]` table, built with `poetry-core` 2.
  Runtime dependencies now declare lower bounds only; `scikit-learn` is declared
  explicitly (it was previously an undeclared transitive dependency).
- `CoxPartialLikelihood` uses a least-squares Newton step when the Hessian is singular
  (no events, constant features, more features than samples, perfect separation).
- `CoxRidge` now maximises a genuinely L2-penalised partial likelihood instead of
  shrinking the unpenalised solution; `CoxLasso` uses proximal gradient descent on
  the L1-penalised partial likelihood. Both objectives are normalised by the number
  of samples, so `alpha` is comparable across dataset sizes.
- `SurvivalDataset` accepts `validate=False` so raw data with missing or invalid
  entries can be passed to the preprocessing utilities.
- `train_test_split_survival` honours absolute `train_size` and `test_size` values
  when stratifying.
- `DRLCoxEstimator` validates inputs with `sklearn.utils.validation.validate_data`
  (scikit-learn >= 1.6); feature-count mismatches raise scikit-learn's standard error.
- The package version is read from installed metadata (`importlib.metadata`).

### Fixed

- `DRLCoxResult.s` is now aligned with the rows of the input data (it was returned in
  descending-time order) and is zero for censored observations.
- `fit_drl_cox` failed on NumPy >= 2.5 (`float()` of a one-element array).
- `DRLCoxEstimator.fit` failed on scikit-learn >= 1.7 (removed private `_validate_data`).
- IPCW weights in `SurvivalStandardScaler` could become infinite when the censoring
  survival estimate reached zero.
- Undefined `SurvivalDataset` name in `drl_cox.plotting`.
- Numerous test defects that had kept CI red since the repository was created
  (wrong concordance orientation, wrong coefficient sign under perfect separation,
  stale regression values, expectations that contradicted `SurvivalDataset`
  validation).

### Added

- `.zenodo.json` so that GitHub releases archived on Zenodo carry the right title,
  author, license, keywords and a link to the DRL-Cox paper.
- `fit_drl_cox(formulation=...)`: `"events"` builds slack constraints only for event
  rows (same optimum, about half the problem size); `"auto"` (default) uses it for ECOS
  and SCS and keeps the full formulation for Clarabel, which stalls on the reduced one.
  The formulation used is reported in `DRLCoxResult.info`.
- CI matrix on Linux (Python 3.11-3.13), macOS and Windows, with lint, format,
  type-check, lock consistency, package build verification and strict docs build.
- Release workflow: tag `vX.Y.Z` publishes to PyPI with trusted publishing and
  creates a GitHub release.
- MkDocs Material documentation with an auto-generated API reference, deployed to
  GitHub Pages.
- `CITATION.cff`, `SECURITY.md`, `CODE_OF_CONDUCT.md`, issue forms, `.editorconfig`,
  `.gitattributes`, and grouped Dependabot updates.
- `py.typed` marker; the package is type-checked with mypy in strict mode.

### Removed

- The weekly "auto upgrade pyproject constraints" workflow, the "update poetry lock"
  workflow and `scripts/pyproject_updater.py`. Bumping lower bounds to the latest
  releases every week is not appropriate for a library and repeatedly broke the
  lock file; Dependabot keeps the lock current instead.
- Import-time dependency probing in `drl_cox/__init__.py` (all dependencies are
  declared).
- Stray `init.py` files and an empty tutorial notebook.

## [0.1.0] - 2025-10-12

### Added

- Initial implementation of the DRL-Cox solver with CVXPY.
- Baseline Cox models (partial likelihood, ridge, lasso).
- Survival metrics (C-index, time-dependent iAUC/IPCW).
- Data loading utilities for WHAS500-style datasets.
- Contamination utilities (covariate shift, outliers).
- Cross-validation framework for epsilon tuning.
- Demo script for the WHAS500 dataset.

[Unreleased]: https://github.com/DiogoRibeiro7/drl-cox/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/DiogoRibeiro7/drl-cox/releases/tag/v0.1.0
