from __future__ import annotations

import numpy as np
import pytest

from drl_cox import (
    concordance_index,
    fit_drl_cox,
    risk_linear_predictor,
    simulate_cox_data,
)
from drl_cox.preprocessing import SurvivalStandardScaler, train_test_split_survival


@pytest.mark.filterwarnings("ignore:Solution may be inaccurate")
def test_survival_workflow_end_to_end():
    """Full pipeline: preprocess -> fit -> evaluate."""
    data = simulate_cox_data(n=90, d=6, seed=314)
    scaler = SurvivalStandardScaler()
    train_data, test_data = train_test_split_survival(
        data,
        test_size=0.3,
        stratify_by="event",
        random_state=314,
    )

    train_scaled = scaler.fit_transform(train_data)
    test_scaled = scaler.transform(test_data)

    result = fit_drl_cox(
        train_scaled,
        epsilon=0.05,
        p=2.0,
        gamma=2,
        solver="SCS",
        solver_opts={"max_iters": 200},
    )

    assert result.status in {"optimal", "optimal_inaccurate"}

    risk_scores = risk_linear_predictor(test_scaled.X, result.beta)
    score = concordance_index(risk_scores, test_scaled.y, test_scaled.zeta)

    assert risk_scores.shape == (test_scaled.X.shape[0],)
    assert 0.0 <= score <= 1.0
    # Expect better than random discrimination on this synthetic data
    assert score > 0.55


@pytest.mark.filterwarnings("ignore:Solution may be inaccurate")
def test_fit_drl_cox_regression_values():
    """Regression guard: coefficients should remain stable for fixed seed."""
    data = simulate_cox_data(n=50, d=5, seed=2024)

    result = fit_drl_cox(
        data,
        epsilon=0.05,
        p=2.0,
        gamma=2,
        solver="CLARABEL",
    )

    # Reference values agree across CLARABEL, ECOS and SCS (with tight tolerances).
    expected_beta = np.array([-0.333013, 1.074195, 0.251388, 0.060238, -0.555921])
    expected_alpha = 0.274836
    expected_objective = 1.306351

    np.testing.assert_allclose(result.beta, expected_beta, atol=1e-3, rtol=5e-3)
    assert pytest.approx(expected_alpha, abs=1e-3, rel=5e-3) == result.alpha
    assert pytest.approx(expected_objective, abs=1e-3, rel=5e-3) == result.objective_value
