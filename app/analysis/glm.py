"""KALESS Engine — General Linear Model (GLM) Module.

Implements Univariate ANOVA / ANCOVA via statsmodels OLS:
  - Fixed Factors (categorical predictors)
  - Covariates (continuous predictors)
  - Type III Sum of Squares
  - Tests of Between-Subjects Effects table (SPSS format)
"""

from __future__ import annotations

import time
from datetime import datetime

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols as sm_ols
from scipy import stats

from app.core.preprocessing import validate_variable_exists
from app.schemas.results import (
    NormalizedResult,
    OutputBlock,
    OutputBlockType,
    Interpretation,
)
from app.utils.errors import ValidationError


def run_glm_univariate(
    df: pd.DataFrame,
    dependent: str,
    fixed_factors: list[str],
    covariates: list[str] | None = None,
) -> NormalizedResult:
    """Run Univariate GLM (ANOVA / ANCOVA).

    Args:
        df: The dataset.
        dependent: Dependent variable (continuous/scale).
        fixed_factors: List of categorical fixed factor variables.
        covariates: Optional list of continuous covariate variables.

    Returns:
        NormalizedResult with Tests of Between-Subjects Effects and descriptives.
    """
    start = time.time()
    warnings: list[str] = []
    covariates = covariates or []

    # Validate
    validate_variable_exists(df, dependent)
    for f in fixed_factors:
        validate_variable_exists(df, f)
    for c in covariates:
        validate_variable_exists(df, c)

    all_vars = [dependent] + fixed_factors + covariates
    df_clean = df[all_vars].dropna()
    n = len(df_clean)
    n_excluded = len(df) - n

    if n < len(all_vars) + 1:
        raise ValidationError(
            f"Not enough valid cases ({n}) for the model. "
            f"Need at least {len(all_vars) + 1}."
        )

    if n_excluded > 0:
        warnings.append(f"{n_excluded} case(s) excluded due to missing values (listwise deletion).")

    # Build formula: dependent ~ C(factor1) * C(factor2) + covariate1 + covariate2
    # For SPSS parity, we use main effects and interactions for fixed factors
    factor_terms = [f"C({f})" for f in fixed_factors]

    # Main effects + all 2-way interactions for factors
    formula_parts = []
    for ft in factor_terms:
        formula_parts.append(ft)

    # Two-way interactions
    if len(factor_terms) > 1:
        for i in range(len(factor_terms)):
            for j in range(i + 1, len(factor_terms)):
                formula_parts.append(f"{factor_terms[i]}:{factor_terms[j]}")

    # Add covariates
    for c in covariates:
        formula_parts.append(c)

    formula_str = f"{dependent} ~ {' + '.join(formula_parts)}"

    # Fit the OLS model
    try:
        model = sm_ols(formula_str, data=df_clean).fit()
    except Exception as e:
        raise ValidationError(f"Model fitting failed: {str(e)}")

    # Type III ANOVA table
    try:
        anova_table = sm.stats.anova_lm(model, typ=3)
    except Exception:
        # Fallback to Type I
        anova_table = sm.stats.anova_lm(model, typ=1)
        warnings.append("Type III SS failed; falling back to Type I.")

    # Build SPSS-style "Tests of Between-Subjects Effects" table
    output_blocks = []

    # Grand stats
    ss_total_corrected = float(anova_table["sum_sq"].sum())
    df_total = int(anova_table["df"].sum())
    grand_mean = float(df_clean[dependent].mean())
    ss_corrected_model = ss_total_corrected - float(anova_table.loc["Residual", "sum_sq"])
    df_corrected_model = df_total - int(anova_table.loc["Residual", "df"])

    rows = []

    # Corrected Model row
    ms_model = ss_corrected_model / df_corrected_model if df_corrected_model > 0 else 0
    ms_error = float(anova_table.loc["Residual", "sum_sq"]) / float(anova_table.loc["Residual", "df"]) if float(anova_table.loc["Residual", "df"]) > 0 else 1
    f_model = ms_model / ms_error if ms_error > 0 else 0
    p_model = 1 - stats.f.cdf(f_model, df_corrected_model, int(anova_table.loc["Residual", "df"]))

    rows.append({
        "Source": "Corrected Model",
        "Type III Sum of Squares": round(ss_corrected_model, 3),
        "df": df_corrected_model,
        "Mean Square": round(ms_model, 3),
        "F": round(f_model, 3),
        "Sig.": f"{p_model:.3f}" if p_model >= 0.0005 else "< .001",
    })

    # Intercept row
    if "Intercept" in anova_table.index:
        ss_int = float(anova_table.loc["Intercept", "sum_sq"])
        df_int = int(anova_table.loc["Intercept", "df"])
        ms_int = ss_int / df_int if df_int > 0 else 0
        f_int = ms_int / ms_error if ms_error > 0 else 0
        p_int = 1 - stats.f.cdf(f_int, df_int, int(anova_table.loc["Residual", "df"]))
        rows.append({
            "Source": "Intercept",
            "Type III Sum of Squares": round(ss_int, 3),
            "df": df_int,
            "Mean Square": round(ms_int, 3),
            "F": round(f_int, 3),
            "Sig.": f"{p_int:.3f}" if p_int >= 0.0005 else "< .001",
        })

    # Individual effect rows (factors, covariates, interactions)
    for idx_name in anova_table.index:
        if idx_name in ("Residual", "Intercept"):
            continue

        ss = float(anova_table.loc[idx_name, "sum_sq"])
        df_eff = int(anova_table.loc[idx_name, "df"])
        ms_eff = ss / df_eff if df_eff > 0 else 0
        f_eff = ms_eff / ms_error if ms_error > 0 else 0
        p_eff = 1 - stats.f.cdf(f_eff, df_eff, int(anova_table.loc["Residual", "df"])) if df_eff > 0 else 1.0

        # Clean up source name: C(gender) → gender
        source_name = str(idx_name).replace("C(", "").replace(")", "").replace(":", " * ")

        rows.append({
            "Source": source_name,
            "Type III Sum of Squares": round(ss, 3),
            "df": df_eff,
            "Mean Square": round(ms_eff, 3),
            "F": round(f_eff, 3),
            "Sig.": f"{p_eff:.3f}" if p_eff >= 0.0005 else "< .001",
        })

    # Error row
    ss_error = float(anova_table.loc["Residual", "sum_sq"])
    df_error = int(anova_table.loc["Residual", "df"])
    rows.append({
        "Source": "Error",
        "Type III Sum of Squares": round(ss_error, 3),
        "df": df_error,
        "Mean Square": round(ms_error, 3),
        "F": "",
        "Sig.": "",
    })

    # Total row
    rows.append({
        "Source": "Total",
        "Type III Sum of Squares": round(float(df_clean[dependent].pow(2).sum()), 3),
        "df": n,
        "Mean Square": "",
        "F": "",
        "Sig.": "",
    })

    # Corrected Total row
    ss_corrected_total = float(df_clean[dependent].var(ddof=1) * (n - 1))
    rows.append({
        "Source": "Corrected Total",
        "Type III Sum of Squares": round(ss_corrected_total, 3),
        "df": n - 1,
        "Mean Square": "",
        "F": "",
        "Sig.": "",
    })

    output_blocks.append(OutputBlock(
        block_type=OutputBlockType.TABLE,
        title="Tests of Between-Subjects Effects",
        display_order=1,
        content={
            "columns": ["Source", "Type III Sum of Squares", "df", "Mean Square", "F", "Sig."],
            "rows": rows,
            "footnotes": [
                f"Dependent Variable: {dependent}",
                f"R Squared = {round(model.rsquared, 3)} (Adjusted R Squared = {round(model.rsquared_adj, 3)})",
            ],
        },
    ))

    # Descriptive Statistics block
    desc_rows = []
    for name, group in df_clean.groupby(fixed_factors if len(fixed_factors) == 1 else fixed_factors[0]):
        desc_rows.append({
            fixed_factors[0]: str(name),
            "Mean": round(float(group[dependent].mean()), 4),
            "Std. Deviation": round(float(group[dependent].std(ddof=1)), 4),
            "N": len(group),
        })
    desc_rows.append({
        fixed_factors[0]: "Total",
        "Mean": round(float(df_clean[dependent].mean()), 4),
        "Std. Deviation": round(float(df_clean[dependent].std(ddof=1)), 4),
        "N": n,
    })

    output_blocks.insert(0, OutputBlock(
        block_type=OutputBlockType.TABLE,
        title="Descriptive Statistics",
        display_order=0,
        content={
            "columns": [fixed_factors[0], "Mean", "Std. Deviation", "N"],
            "rows": desc_rows,
            "footnotes": [f"Dependent Variable: {dependent}"],
        },
    ))

    # R-squared
    r2 = round(model.rsquared, 3)
    adj_r2 = round(model.rsquared_adj, 3)

    duration = int((time.time() - start) * 1000)

    return NormalizedResult(
        analysis_type="glm_univariate",
        title="Univariate Analysis of Variance",
        variables={
            "dependent": [dependent],
            "fixed_factors": fixed_factors,
            "covariates": covariates,
        },
        model_summary={"R_squared": r2, "Adjusted_R_squared": adj_r2},
        output_blocks=output_blocks,
        interpretation=Interpretation(
            summary=(
                f"A General Linear Model (Univariate) was conducted with {dependent} as the dependent variable. "
                f"R² = {r2}, Adjusted R² = {adj_r2}."
            ),
            academic_sentence=(
                f"A {'two' if len(fixed_factors) > 1 else 'one'}-way "
                f"{'ANCOVA' if covariates else 'ANOVA'} was conducted. "
                f"The model explained {round(r2 * 100, 1)}% of the variance in {dependent} "
                f"(R² = {r2}, Adjusted R² = {adj_r2})."
            ),
            recommendations=[],
        ),
        warnings=warnings,
        metadata={
            "n_total": len(df),
            "missing_excluded": n_excluded,
            "valid_n": n,
            "formula": formula_str,
            "library": "statsmodels.formula.api.ols",
            "duration_ms": duration,
            "timestamp": datetime.utcnow().isoformat(),
        },
    )
