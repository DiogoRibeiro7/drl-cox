# DRL-Cox

Distributionally robust Cox regression with Wasserstein ambiguity sets, in Python.

`drl-cox` implements the **DRL-Cox** model introduced by Yeping Jin, Lauren Wise and
Ioannis Ch. Paschalidis in *Distributionally Robust Learning in Survival Analysis*
(CHIL 2025, PMLR 287; [arXiv:2506.01348](https://arxiv.org/abs/2506.01348)).
The Cox partial likelihood is replaced by a worst-case objective over a Wasserstein
ball around the empirical distribution, which yields a tractable exponential-cone
program that is more robust to outliers, covariate shift and model misspecification.

## What is included

- **Solver**: `fit_drl_cox` builds and solves the exponential-cone program with CVXPY
  (Clarabel by default).
- **Baselines**: Cox partial likelihood, ridge and lasso.
- **Metrics**: Harrell's C-index and IPCW time-dependent AUC / iAUC.
- **Model selection**: sequential or parallel cross-validation over the Wasserstein
  radius, and Bayesian optimisation with scikit-optimize.
- **Data utilities**: synthetic data, CSV loading, contamination injectors and
  survival-aware preprocessing.
- **scikit-learn API**: `DRLCoxEstimator` for pipelines and grid searches.

## Where to start

- [Installation](installation.md)
- [Quick start](quickstart.md)
- [Examples](examples.md)
- [scikit-learn API](sklearn_api.md)
- [API reference](reference.md)

## Citing

If you use this package, cite both the software and the paper it implements.
Releases are archived on Zenodo; the concept DOI
[10.5281/zenodo.22722209](https://doi.org/10.5281/zenodo.22722209) always resolves to the latest version.

```bibtex
@software{ribeiro_drl_cox,
  title     = {drl-cox: distributionally robust Cox regression in Python},
  author    = {Ribeiro, Diogo},
  year      = {2026},
  doi       = {10.5281/zenodo.22722209},
  url       = {https://github.com/DiogoRibeiro7/drl-cox}
}
```

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
