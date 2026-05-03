"""KALESS Engine — Linear Regression Module (statsmodels OLS)."""

import time
from datetime import datetime
import pandas as pd
import numpy as np
import statsmodels.api as sm

from app.core.preprocessing import validate_variable_exists
from app.schemas.results import NormalizedResult, OutputBlock, OutputBlockType

def run_linear_regression(df: pd.DataFrame, dependent: str, independents: list[str]) -> NormalizedResult:
    """Computes Linear Regression with SPSS-style output including the Holy Trinity tables."""
    start = time.time()
    warnings: list[str] = []

    validate_variable_exists(df, dependent)
    for var in independents:
        validate_variable_exists(df, var)

    # Prepare data, drop missing values listwise
    vars_to_keep = [dependent] + independents
    sub_df = df[vars_to_keep].dropna()
    valid_n = len(sub_df)
    
    if valid_n < len(independents) + 2:
        raise ValueError("Not enough valid cases to compute Linear Regression.")

    if len(sub_df) < len(df):
        warnings.append(f"{len(df) - valid_n} case(s) excluded due to missing values.")

    # Variables
    y = sub_df[dependent]
    X = sub_df[independents]

    # Add constant for OLS (CRITICAL — ensures the intercept is modeled)
    X_with_const = sm.add_constant(X)

    # Fit the model
    model = sm.OLS(y, X_with_const)
    results = model.fit()

    # ═══ TABLE 1: Model Summary ═══
    r_squared = results.rsquared
    adj_r_squared = results.rsquared_adj
    r_value = np.sqrt(r_squared) if r_squared > 0 else 0
    std_err_estimate = np.sqrt(results.scale)
    
    f_change = results.fvalue
    sig_f_change = results.f_pvalue

    model_summary_rows = [{
        "Model": "1",
        "R": f"{r_value:.3f}",
        "R Square": f"{r_squared:.3f}",
        "Adjusted R Square": f"{adj_r_squared:.3f}",
        "Std. Error of the Estimate": f"{std_err_estimate:.5f}",
        "R Square Change": f"{r_squared:.3f}",
        "F Change": f"{f_change:.3f}",
        "df1": str(int(results.df_model)),
        "df2": str(int(results.df_resid)),
        "Sig. F Change": f"{sig_f_change:.3f}"
    }]

    # ═══ TABLE 2: ANOVA ═══
    df_model = int(results.df_model)
    df_resid = int(results.df_resid)
    df_total = df_model + df_resid
    ss_total = results.centered_tss
    ss_resid = results.ssr
    ss_model = ss_total - ss_resid
    ms_model = ss_model / df_model if df_model > 0 else 0
    ms_resid = ss_resid / df_resid if df_resid > 0 else 0

    anova_rows = [
        {"Model": "Regression", "Sum of Squares": f"{ss_model:.3f}", "df": str(df_model), "Mean Square": f"{ms_model:.3f}", "F": f"{results.fvalue:.3f}", "Sig.": f"{results.f_pvalue:.3f}"},
        {"Model": "Residual", "Sum of Squares": f"{ss_resid:.3f}", "df": str(df_resid), "Mean Square": f"{ms_resid:.3f}", "F": "", "Sig.": ""},
        {"Model": "Total", "Sum of Squares": f"{ss_total:.3f}", "df": str(df_total), "Mean Square": "", "F": "", "Sig.": ""}
    ]

    # ═══ TABLE 3: Coefficients ═══
    # Calculate standardized betas (Beta) by running OLS on z-scored data without intercept
    y_std = (y - y.mean()) / y.std(ddof=1)
    X_std = (X - X.mean()) / X.std(ddof=1)
    try:
        std_results = sm.OLS(y_std, X_std).fit()
        std_betas = std_results.params
    except Exception:
        std_betas = {}

    coefficients_rows = []
    for var in ["const"] + independents:
        display_name = "(Constant)" if var == "const" else var
        b = results.params[var]
        se = results.bse[var]
        beta = std_betas.get(var, None) if var != "const" else None
        t_val = results.tvalues[var]
        sig = results.pvalues[var]
        
        coefficients_rows.append({
            "Model": display_name,
            "B": f"{b:.3f}",
            "Std. Error": f"{se:.3f}",
            "Beta": f"{beta:.3f}" if beta is not None else "",
            "t": f"{t_val:.3f}",
            "Sig.": f"{sig:.3f}"
        })

    duration = int((time.time() - start) * 1000)

    output_blocks = [
        OutputBlock(
            block_type=OutputBlockType.TABLE,
            title="Model Summary",
            content={
                "columns": ["Model", "R", "R Square", "Adjusted R Square", "Std. Error of the Estimate", "R Square Change", "F Change", "df1", "df2", "Sig. F Change"],
                "rows": model_summary_rows,
                "footnotes": [f"a. Predictors: (Constant), {', '.join(independents)}"]
            }
        ),
        OutputBlock(
            block_type=OutputBlockType.TABLE,
            title="ANOVA",
            content={
                "columns": ["Model", "Sum of Squares", "df", "Mean Square", "F", "Sig."],
                "rows": anova_rows,
                "footnotes": [
                    f"a. Dependent Variable: {dependent}",
                    f"b. Predictors: (Constant), {', '.join(independents)}"
                ]
            }
        ),
        OutputBlock(
            block_type=OutputBlockType.TABLE,
            title="Coefficients",
            content={
                "columns": ["Model", "B", "Std. Error", "Beta", "t", "Sig."],
                "rows": coefficients_rows,
                "footnotes": [f"a. Dependent Variable: {dependent}"],
                "column_groups": [
                    {"label": "Unstandardized Coefficients", "columns": ["B", "Std. Error"]},
                    {"label": "Standardized Coefficients", "columns": ["Beta"]}
                ]
            }
        )
    ]

    from app.utils.interpretation import generate_interpretation

    res = NormalizedResult(
        analysis_type="linear_regression",
        title="Regression",
        variables={"dependent": [dependent], "independent": independents},
        output_blocks=output_blocks,
        warnings=warnings,
        primary={
            "statistic_name": "F",
            "statistic_value": float(results.fvalue),
            "df": float(results.df_model),
            "df2": float(results.df_resid),
            "p_value": float(results.f_pvalue),
            "p_value_formatted": "p < .001" if results.f_pvalue < 0.001 else f"p = {results.f_pvalue:.3f}",
            "significance": "significant" if results.f_pvalue < 0.05 else "not_significant"
        },
        metadata={
            "valid_n": valid_n, 
            "duration_ms": duration, 
            "timestamp": datetime.utcnow().isoformat(),
            "r_squared": float(r_squared),
            "adj_r_squared": float(adj_r_squared)
        }
    )
    res.interpretation = generate_interpretation(res)
    return res

