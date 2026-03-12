"""KALESS Engine — Regression Module.

Implements simple and multiple linear regression.

DEV NOTE: 
Currently uses direct OLS computation (via `numpy.linalg.inv(X'X) X'y`) and `scipy.stats` 
rather than `statsmodels`. This is a design decision from Phase 4 to reduce large
dependencies when `statsmodels` failed to install reliably.

Outputs supported:
- F-statistic, R², Adjusted R², MSE/RMSE
- Coefficients: b, Standard Error (SE), beta (standardized), t, p-value, 95% Confidence Intervals

Diagnostics supported:
- VIF (Variance Inflation Factor) for multicollinearity
- Durbin-Watson statistic for autocorrelation
- Residual normality check (Shapiro-Wilk)
- Residuals vs Fitted scatter chart

Limitations (compared to future full statsmodels implementation):
- Cook's Distance / robust leverage metrics not yet included.
- Robust standard errors (HC0-HC3) not available.
- Categorical dummy coding must be handled prior to running this module; it assumes 
  numeric input for all predictors.
"""

from __future__ import annotations

import time
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats

from app.core.preprocessing import (
    validate_variable_exists, validate_numeric, validate_min_n,
    drop_missing_listwise, compute_descriptive,
)
from app.core.assumptions import check_normality, build_assumptions_block, AssumptionResult
from app.core.effect_sizes import r_squared_effect
from app.core.interpretation import determine_significance, interpret_regression, format_p
from app.schemas.results import (
    NormalizedResult, PrimaryResult, GroupDescriptive,
    CoefficientRow, ChartData,
)


def run_linear_regression(
    df: pd.DataFrame,
    dependent: str,
    predictors: list[str],
    alpha: float = 0.05,
) -> NormalizedResult:
    """Simple or multiple linear regression (OLS via numpy)."""
    start = time.time()
    warnings: list[str] = []

    # Validate variables
    all_vars = [dependent] + predictors
    for v in all_vars:
        validate_variable_exists(df, v)
        validate_numeric(df[v], v)

    if len(predictors) == 0:
        from app.utils.errors import ValidationError
        raise ValidationError("At least one predictor variable is required.")

    cleaned, n_dropped = drop_missing_listwise(df, all_vars)
    if n_dropped > 0:
        warnings.append(f"{n_dropped} case(s) excluded due to missing values.")

    n = len(cleaned)
    k = len(predictors)
    validate_min_n(n, k + 2, "regression requires n > k + 1")

    y = cleaned[dependent].values.astype(float)
    X_raw = cleaned[predictors].values.astype(float)

    # Add constant (intercept)
    X = np.column_stack([np.ones(n), X_raw])  # [1, x1, x2, ...]
    p_count = X.shape[1]  # k + 1 (with intercept)

    # OLS: beta = (X'X)^-1 X'y
    try:
        XtX_inv = np.linalg.inv(X.T @ X)
    except np.linalg.LinAlgError:
        warnings.append("Singular matrix — predictors may be perfectly collinear.")
        XtX_inv = np.linalg.pinv(X.T @ X)

    beta = XtX_inv @ (X.T @ y)
    y_hat = X @ beta
    residuals = y - y_hat

    # Sums of squares
    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    ss_reg = ss_tot - ss_res

    # R² and adjusted R²
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p_count) if n > p_count else r2

    # MSE
    mse = ss_res / (n - p_count) if n > p_count else ss_res
    rmse = float(np.sqrt(mse))

    # F-statistic
    ms_reg = ss_reg / k if k > 0 else 0
    f_stat = ms_reg / mse if mse > 0 else 0.0
    f_p = 1 - stats.f.cdf(f_stat, k, n - p_count) if n > p_count else 1.0
    sig = determine_significance(float(f_p), alpha)

    # Standard errors of coefficients
    se_beta = np.sqrt(np.diag(XtX_inv) * mse)

    # t-values and p-values for each coefficient
    t_values = beta / se_beta
    p_values = 2 * (1 - stats.t.cdf(np.abs(t_values), n - p_count))

    # Confidence intervals
    t_crit = stats.t.ppf(1 - alpha / 2, n - p_count)
    ci_lower = beta - t_crit * se_beta
    ci_upper = beta + t_crit * se_beta

    # Standardized coefficients (beta weights)
    y_std = float(np.std(y, ddof=1))
    x_stds = np.std(X_raw, axis=0, ddof=1)

    # VIF (variance inflation factor) for multiple regression
    vifs: list[float | None] = [None] * k
    if k > 1:
        for j in range(k):
            other_cols = [i for i in range(k) if i != j]
            X_other = np.column_stack([np.ones(n), X_raw[:, other_cols]])
            try:
                XtX_other_inv = np.linalg.inv(X_other.T @ X_other)
                beta_other = XtX_other_inv @ (X_other.T @ X_raw[:, j])
                y_hat_other = X_other @ beta_other
                ss_res_j = float(np.sum((X_raw[:, j] - y_hat_other) ** 2))
                ss_tot_j = float(np.sum((X_raw[:, j] - np.mean(X_raw[:, j])) ** 2))
                r2_j = 1 - ss_res_j / ss_tot_j if ss_tot_j > 0 else 0
                vifs[j] = round(1 / (1 - r2_j), 2) if r2_j < 1 else None
            except Exception:
                vifs[j] = None

    # Build coefficient table
    coef_names = ["(Constant)"] + list(predictors)
    coefficients: list[CoefficientRow] = []
    for i in range(p_count):
        std_beta = None
        vif_val = None
        if i > 0:  # Not intercept
            j = i - 1
            if y_std > 0 and x_stds[j] > 0:
                std_beta = round(float(beta[i] * x_stds[j] / y_std), 4)
            vif_val = vifs[j]

        coefficients.append(CoefficientRow(
            name=coef_names[i],
            b=round(float(beta[i]), 4),
            se=round(float(se_beta[i]), 4),
            beta=std_beta,
            statistic=round(float(t_values[i]), 4),
            p_value=round(float(p_values[i]), 6),
            ci_lower=round(float(ci_lower[i]), 4),
            ci_upper=round(float(ci_upper[i]), 4),
            vif=vif_val,
        ))

    # Multicollinearity warning
    if k > 1:
        vif_vals = [v for v in vifs if v is not None]
        if any(v > 10 for v in vif_vals):
            warnings.append("High multicollinearity detected (VIF > 10). Consider removing correlated predictors.")

    # Assumptions
    residual_series = pd.Series(residuals)
    assumption_checks = [
        check_normality(residual_series, "Residuals", alpha),
    ]

    # Durbin-Watson (manual)
    dw = float(np.sum(np.diff(residuals) ** 2) / ss_res) if ss_res > 0 else 2.0
    assumption_checks.append(AssumptionResult(
        test_name="Durbin-Watson",
        description="Independence of residuals",
        statistic=round(dw, 4),
        passed=1.5 <= dw <= 2.5,
        note=f"DW = {dw:.4f}. Values near 2 suggest no autocorrelation.",
    ))

    assumptions = build_assumptions_block(assumption_checks)

    # Model summary
    model_summary = {
        "r": round(float(np.sqrt(max(r2, 0))), 4),
        "r_squared": round(r2, 4),
        "adj_r_squared": round(adj_r2, 4),
        "se_estimate": round(rmse, 4),
        "f_statistic": round(f_stat, 4),
        "f_p_value": round(float(f_p), 6),
        "n": n,
        "k": k,
    }

    # Effect size
    effect = r_squared_effect(r2)

    # Descriptives
    descriptives = [GroupDescriptive(**compute_descriptive(cleaned[dependent], dependent))]
    for p_name in predictors:
        descriptives.append(GroupDescriptive(**compute_descriptive(cleaned[p_name], p_name)))

    # Charts
    residual_data = [
        {"fitted": round(float(y_hat[i]), 4), "residual": round(float(residuals[i]), 4)}
        for i in range(min(n, 500))
    ]
    charts = [ChartData(
        chart_type="scatter",
        data=residual_data,
        config={"title": "Residuals vs Fitted", "xLabel": "Fitted Values", "yLabel": "Residuals"},
    )]

    if k == 1:
        scatter_data = [
            {predictors[0]: round(float(X_raw[i, 0]), 4), dependent: round(float(y[i]), 4)}
            for i in range(min(n, 500))
        ]
        charts.append(ChartData(
            chart_type="scatter_with_line",
            data=scatter_data,
            config={
                "title": f"{dependent} vs {predictors[0]}",
                "xLabel": predictors[0], "yLabel": dependent,
                "slope": round(float(beta[1]), 4),
                "intercept": round(float(beta[0]), 4),
            },
        ))

    duration = int((time.time() - start) * 1000)

    return NormalizedResult(
        analysis_type="linear_regression",
        title=(f"Simple Linear Regression — {dependent} ~ {predictors[0]}"
               if k == 1
               else f"Multiple Linear Regression — {dependent}"),
        variables={"dependent": dependent, "predictors": predictors},
        assumptions=assumptions,
        primary=PrimaryResult(
            statistic_name="F",
            statistic_value=round(f_stat, 4),
            df=float(k),
            df2=float(n - p_count),
            p_value=round(float(f_p), 6),
            p_value_formatted=format_p(float(f_p)),
            significance=sig,
            alpha=alpha,
        ),
        coefficients=coefficients,
        model_summary=model_summary,
        descriptives=descriptives,
        effect_size=effect,
        charts=charts,
        interpretation=interpret_regression(
            r2, adj_r2, f_stat, k, n - p_count, float(f_p), sig,
        ),
        warnings=warnings,
        metadata={
            "n_total": len(df), "missing_excluded": n_dropped,
            "library": "scipy + numpy (OLS)", "duration_ms": duration,
            "timestamp": datetime.utcnow().isoformat(),
        },
    )
