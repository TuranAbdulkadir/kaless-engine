"""KALESS Engine — Generalized Linear Models Module.

Implements Generalized Linear Models (GLM) and Generalized Estimating Equations (GEE).
"""

from __future__ import annotations

import time
from datetime import datetime

import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from app.core.preprocessing import validate_variable_exists
from app.schemas.results import NormalizedResult, OutputBlock, OutputBlockType

def run_genlin(df: pd.DataFrame, dependent: str, predictors: list[str], family: str = "gaussian") -> NormalizedResult:
    """Run Generalized Linear Model."""
    start = time.time()
    
    validate_variable_exists(df, dependent)
    for p in predictors:
        validate_variable_exists(df, p)
        
    df_clean = df[[dependent] + predictors].dropna()
    
    formula = f"{dependent} ~ {' + '.join(predictors)}" if predictors else f"{dependent} ~ 1"
    
    # Map family string to statsmodels family
    if family == "binomial":
        fam = sm.families.Binomial()
    elif family == "poisson":
        fam = sm.families.Poisson()
    elif family == "gamma":
        fam = sm.families.Gamma()
    else:
        fam = sm.families.Gaussian()
        
    try:
        model = smf.glm(formula=formula, data=df_clean, family=fam)
        result = model.fit()
    except Exception as e:
        raise ValueError(f"Generalized Linear Model failed: {str(e)}")
        
    # Parameter Estimates
    summary_df = result.summary2().tables[1]
    summary_df = summary_df.reset_index().rename(columns={"index": "Parameter"})
    
    rows = []
    for idx, row in summary_df.iterrows():
        rows.append({
            "Parameter": str(row["Parameter"]),
            "B": round(row["Coef."], 3),
            "Std. Error": round(row["Std.Err."], 3),
            "z": round(row["z"], 3),
            "Sig.": f"{row['P>|z|']:.3f}" if row['P>|z|'] >= 0.001 else "< .001",
            "Lower": round(row["[0.025"], 3),
            "Upper": round(row["0.975]"], 3),
        })
        
    output_blocks = [
        OutputBlock(
            block_type=OutputBlockType.TABLE,
            title="Parameter Estimates",
            content={
                "columns": ["Parameter", "B", "Std. Error", "z", "Sig.", "Lower", "Upper"],
                "rows": rows
            }
        ),
        OutputBlock(
            block_type=OutputBlockType.TABLE,
            title="Goodness of Fit",
            content={
                "columns": ["Criterion", "Value"],
                "rows": [
                    {"Criterion": "Deviance", "Value": round(result.deviance, 3)},
                    {"Criterion": "Pearson Chi-Square", "Value": round(result.pearson_chi2, 3)},
                    {"Criterion": "AIC", "Value": round(result.aic, 3) if not pd.isna(result.aic) else "N/A"},
                    {"Criterion": "BIC", "Value": round(result.bic_llf, 3) if hasattr(result, "bic_llf") else "N/A"}
                ]
            }
        )
    ]
    
    return NormalizedResult(
        analysis_type="genlin",
        title="Generalized Linear Models",
        variables={"dependent": dependent, "predictors": predictors},
        output_blocks=output_blocks,
        metadata={
            "n_total": len(df),
            "library": "statsmodels.formula.api.glm",
            "duration_ms": int((time.time() - start) * 1000),
            "timestamp": datetime.utcnow().isoformat(),
        }
    )

def run_gee(df: pd.DataFrame, dependent: str, predictors: list[str], subject: str, family: str = "gaussian") -> NormalizedResult:
    """Run Generalized Estimating Equations."""
    start = time.time()
    
    validate_variable_exists(df, dependent)
    validate_variable_exists(df, subject)
    for p in predictors:
        validate_variable_exists(df, p)
        
    df_clean = df[[dependent, subject] + predictors].dropna()
    groups = df_clean[subject]
    
    formula = f"{dependent} ~ {' + '.join(predictors)}" if predictors else f"{dependent} ~ 1"
    
    if family == "binomial":
        fam = sm.families.Binomial()
    elif family == "poisson":
        fam = sm.families.Poisson()
    else:
        fam = sm.families.Gaussian()
        
    cov_struct = sm.cov_struct.Exchangeable()
        
    try:
        model = smf.gee(formula, data=df_clean, groups=groups, family=fam, cov_struct=cov_struct)
        result = model.fit()
    except Exception as e:
        raise ValueError(f"GEE failed: {str(e)}")
        
    # Parameter Estimates
    summary_df = result.summary2().tables[1]
    summary_df = summary_df.reset_index().rename(columns={"index": "Parameter"})
    
    rows = []
    for idx, row in summary_df.iterrows():
        rows.append({
            "Parameter": str(row["Parameter"]),
            "B": round(row["Coef."], 3),
            "Std. Error": round(row["Std.Err."], 3),
            "z": round(row["z"], 3),
            "Sig.": f"{row['P>|z|']:.3f}" if row['P>|z|'] >= 0.001 else "< .001",
            "Lower": round(row["[0.025"], 3),
            "Upper": round(row["0.975]"], 3),
        })
        
    output_blocks = [
        OutputBlock(
            block_type=OutputBlockType.TABLE,
            title="Parameter Estimates (GEE)",
            content={
                "columns": ["Parameter", "B", "Std. Error", "z", "Sig.", "Lower", "Upper"],
                "rows": rows
            }
        )
    ]
    
    return NormalizedResult(
        analysis_type="gee",
        title="Generalized Estimating Equations",
        variables={"dependent": dependent, "predictors": predictors, "subject": subject},
        output_blocks=output_blocks,
        metadata={
            "n_total": len(df),
            "library": "statsmodels.formula.api.gee",
            "duration_ms": int((time.time() - start) * 1000),
            "timestamp": datetime.utcnow().isoformat(),
        }
    )
