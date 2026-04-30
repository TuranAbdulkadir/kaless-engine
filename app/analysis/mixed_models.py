"""Mixed Models (Linear Mixed-Effects Model)."""

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from app.schemas.results import NormalizedResult, OutputBlock

def run_mixed_model(df: pd.DataFrame, dependent: str, fixed_factors: list[str], random_factor: str) -> NormalizedResult:
    """Run Linear Mixed-Effects Model via statsmodels MixedLM."""
    all_cols = [dependent] + fixed_factors + [random_factor]
    for col in all_cols:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in dataset.")

    df_clean = df.dropna(subset=all_cols).copy()

    if len(df_clean) < 10:
        raise ValueError("Mixed models require at least 10 valid cases.")

    # Build formula: dependent ~ fixed1 + fixed2 + ...
    formula = f"{dependent} ~ " + " + ".join(fixed_factors)

    try:
        model = smf.mixedlm(formula, df_clean, groups=df_clean[random_factor])
        result = model.fit(reml=True)
    except Exception as e:
        raise ValueError(f"Mixed model failed to converge: {str(e)}")

    # Fixed Effects table
    fe_rows = []
    for param_name in result.fe_params.index:
        fe_rows.append({
            "Effect": str(param_name),
            "Estimate": f"{result.fe_params[param_name]:.4f}",
            "Std. Error": f"{result.bse_fe[param_name]:.4f}",
            "z": f"{result.tvalues[param_name]:.3f}",
            "Sig.": f"{result.pvalues[param_name]:.4f}"
        })

    # Random Effects variance
    re_rows = [
        {"Parameter": "Group Var", "Estimate": f"{result.cov_re.iloc[0, 0]:.4f}"}
    ]

    output_blocks = [
        OutputBlock(
            block_type=OutputBlockType.TABLE,
            title="Type III Tests of Fixed Effects",
            content={
                "columns": ["Effect", "Estimate", "Std. Error", "z", "Sig."],
                "rows": fe_rows,
                "footnotes": ["Dependent Variable: " + dependent]
            }
        ),
        OutputBlock(
            block_type=OutputBlockType.TABLE,
            title="Estimates of Covariance Parameters",
            content={
                "columns": ["Parameter", "Estimate"],
                "rows": re_rows,
                "footnotes": [f"Random Effect grouping: {random_factor}"]
            }
        ),
        OutputBlock(
            block_type=OutputBlockType.TABLE,
            title="Model Fit Information",
            content={
                "columns": ["Metric", "Value"],
                "rows": [
                    {"Metric": "Log-Likelihood", "Value": f"{result.llf:.3f}"},
                    {"Metric": "Converged", "Value": str(result.converged)},
                    {"Metric": "Method", "Value": "REML"},
                ],
                "footnotes": []
            }
        )
    ]

    return NormalizedResult(
        title="Linear Mixed Model",
        variables={"analyzed": all_cols},
        output_blocks=output_blocks,
        interpretation={"academic_sentence": f"A linear mixed-effects model was fitted with {', '.join(fixed_factors)} as fixed effects and {random_factor} as a random effect."}
    )
