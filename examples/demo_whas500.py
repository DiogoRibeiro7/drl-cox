from __future__ import annotations

import os

from drl_cox import (
    CoxLasso,
    CoxPartialLikelihood,
    CoxRidge,
    SurvivalDataset,
    concordance_index,
    cross_validate_epsilon,
    fit_drl_cox,
    inject_covariate_shift,
    inject_outliers,
    load_whas500_like_csv,
    risk_linear_predictor,
    simulate_cox_data,
    time_dependent_auc_iAUC,
)

CSV_PATH = os.environ.get("WHAS500_CSV", "whas500_sample.csv")

try:
    data = load_whas500_like_csv(CSV_PATH)
    print(f"Loaded CSV from {CSV_PATH} with X={data.X.shape}")
except Exception as e:
    print(f"Could not load '{CSV_PATH}' ({e}); using synthetic demo data.")
    data = simulate_cox_data(n=300, d=8, seed=123, baseline_hazard=0.02, censor_rate=0.5)

# Induce some contamination to mimic paper experiments
X_shifted = inject_covariate_shift(data.X, feature_indices=[0, 1], mean=0.0, std=2.0, seed=1)
X_noisy = inject_outliers(X_shifted, ratio=0.2, severity_std=2.5, seed=2)
contaminated = SurvivalDataset(X=X_noisy, y=data.y, zeta=data.zeta)

# --- DRL-Cox: tune epsilon via CV ---
print("Cross-validating epsilon (C-index)...")
cv = cross_validate_epsilon(
    contaminated,
    epsilons=[0.0, 0.02, 0.05, 0.1, 0.2],
    p=2.0,
    gamma=3,
    kfolds=3,
    metric="cindex",
    solver="CLARABEL",
    solver_opts={"max_iter": 200},
)
print(cv.groupby("epsilon")["score"].mean())

best_eps = float(cv.groupby("epsilon")["score"].mean().idxmax())
print(f"Best epsilon: {best_eps}")

res = fit_drl_cox(
    contaminated, epsilon=best_eps, p=2.0, gamma=3, solver="CLARABEL", solver_opts={"max_iter": 300}
)
rb = res.beta
r_scores = risk_linear_predictor(contaminated.X, rb)

cidx = concordance_index(r_scores, contaminated.y, contaminated.zeta)
iauc = time_dependent_auc_iAUC(r_scores, contaminated.y, contaminated.zeta)
print(f"DRL-Cox  C-index={cidx:.3f}  iAUC={iauc:.3f}")

# --- Baselines ---
cox_pl = CoxPartialLikelihood().fit(contaminated.X, contaminated.y, contaminated.zeta)
cox_ridge = CoxRidge(alpha=1.0).fit(contaminated.X, contaminated.y, contaminated.zeta)
cox_lasso = CoxLasso(alpha=0.02).fit(contaminated.X, contaminated.y, contaminated.zeta)

for name, beta in [
    ("Cox-PL", cox_pl),
    ("Cox-Ridge", cox_ridge),
    ("Cox-Lasso", cox_lasso),
]:
    rs = risk_linear_predictor(contaminated.X, beta)
    print(
        f"{name:10s}  C-index={concordance_index(rs, contaminated.y, contaminated.zeta):.3f}  "
        f"iAUC={time_dependent_auc_iAUC(rs, contaminated.y, contaminated.zeta):.3f}"
    )
