"""KALESS Engine — Correlation Module."""

import time
from datetime import datetime
import pandas as pd
import numpy as np
from scipy import stats

from app.core.preprocessing import validate_variable_exists, validate_numeric, drop_missing_listwise
from app.schemas.results import NormalizedResult, OutputBlock, OutputBlockType

def calculate_correlation(df: pd.DataFrame, variables: list[str], method: str = "pearson") -> NormalizedResult:
    """Computes a correlation matrix (Pearson or Spearman) in SPSS-style layout."""
    start = time.time()
    warnings = []
    
    try:
        if len(variables) < 2:
            raise ValueError("Correlation requires at least 2 variables.")
            
        for v in variables:
            validate_variable_exists(df, v)
            validate_numeric(df[v], v)
            
        cleaned, n_dropped = drop_missing_listwise(df, variables)
        if n_dropped > 0:
            warnings.append(f"{n_dropped} missing cases excluded.")
            
        if len(cleaned) < 2:
            raise ValueError("Not enough valid cases to compute correlation.")
            
        n_valid = len(cleaned)
        
        # Compute correlation matrix
        corr_matrix_data = []
        for v1 in variables:
            row_results = {"Variable": v1}
            for v2 in variables:
                if v1 == v2:
                    row_results[v2] = "1"
                    row_results[f"{v2}_sig"] = "-"
                    row_results[f"{v2}_N"] = str(n_valid)
                else:
                    if method == "spearman":
                        r, p = stats.spearmanr(cleaned[v1], cleaned[v2])
                    else:
                        r, p = stats.pearsonr(cleaned[v1], cleaned[v2])
                    
                    row_results[v2] = f"{r:.3f}"
                    if p < 0.01:
                        row_results[v2] += "**"
                    elif p < 0.05:
                        row_results[v2] += "*"
                    row_results[f"{v2}_sig"] = f"{p:.3f}"
                    row_results[f"{v2}_N"] = str(n_valid)
            corr_matrix_data.append(row_results)

        # Output Table Construction
        rows = []
        metric_name = "Pearson Correlation" if method == "pearson" else "Spearman's rho"
        columns = ["Variable", "Metric"] + variables
        for v1 in variables:
            # Row 1: Correlation
            r1 = {"Variable": v1, "Metric": metric_name}
            for v2 in variables:
                r1[v2] = next(item for item in corr_matrix_data if item["Variable"] == v1)[v2]
            rows.append(r1)
            
            # Row 2: Sig (2-tailed)
            r2 = {"Variable": "", "Metric": "Sig. (2-tailed)"}
            for v2 in variables:
                r2[v2] = next(item for item in corr_matrix_data if item["Variable"] == v1)[f"{v2}_sig"]
            rows.append(r2)
            
            # Row 3: N
            r3 = {"Variable": "", "Metric": "N"}
            for v2 in variables:
                r3[v2] = next(item for item in corr_matrix_data if item["Variable"] == v1)[f"{v2}_N"]
            rows.append(r3)

        output_blocks = [
            OutputBlock(
                block_type=OutputBlockType.TABLE,
                title="Correlations",
                content={
                    "columns": columns, 
                    "rows": rows,
                    "footnotes": [
                        "** Correlation is significant at the 0.01 level (2-tailed).",
                        "* Correlation is significant at the 0.05 level (2-tailed)."
                    ]
                }
            )
        ]
        
        from app.utils.interpretation import generate_interpretation

        res = NormalizedResult(
            analysis_type=f"{method}_correlation",
            title=f"{method.capitalize()} Correlations",
            variables={"analyzed": variables},
            output_blocks=output_blocks,
            metadata={
                "n_total": len(df),
                "valid_n": n_valid,
                "duration_ms": int((time.time() - start) * 1000),
                "timestamp": datetime.utcnow().isoformat(),
                "method": method
            },
        )
        res.interpretation = generate_interpretation(res)
        return res
    except Exception as e:
        raise ValueError(f"Correlation failed: {str(e)}")
