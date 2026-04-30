"""Missing Value Analysis."""

import pandas as pd
import numpy as np
from app.schemas.results import NormalizedResult, OutputBlock

def run_missing_value_analysis(df: pd.DataFrame, variables: list[str]) -> NormalizedResult:
    """Run Missing Value Analysis and return patterns."""
    if not variables:
        variables = df.columns.tolist()
        
    df_sub = df[variables]
    n_cases = len(df_sub)
    
    # Univariate Statistics
    univariate_rows = []
    for col in variables:
        n_missing = df_sub[col].isna().sum()
        pct_missing = (n_missing / n_cases) * 100
        n_valid = n_cases - n_missing
        
        row = {
            "Variable": col,
            "N": str(n_valid),
            "Missing Count": str(n_missing),
            "Missing Percent": f"{pct_missing:.1f}%"
        }
        
        # Add basic stats if numeric
        if pd.api.types.is_numeric_dtype(df_sub[col]):
            row["Mean"] = f"{df_sub[col].mean():.3f}"
            row["Std. Deviation"] = f"{df_sub[col].std():.3f}"
        else:
            row["Mean"] = ""
            row["Std. Deviation"] = ""
            
        univariate_rows.append(row)
        
    output_blocks = [
        OutputBlock(
            block_type=OutputBlockType.TABLE,
            title="Univariate Statistics",
            content={
                "columns": ["Variable", "N", "Mean", "Std. Deviation", "Missing Count", "Missing Percent"],
                "rows": univariate_rows,
                "footnotes": []
            }
        )
    ]
    
    # Missing Patterns (Summarized)
    # Count rows with 0 missing, 1 missing, etc.
    missing_counts_per_row = df_sub.isna().sum(axis=1)
    pattern_counts = missing_counts_per_row.value_counts().sort_index()
    
    pattern_rows = []
    for num_missing, count in pattern_counts.items():
        pattern_rows.append({
            "Missing Values": str(num_missing),
            "Number of Cases": str(count),
            "Percent of Total": f"{(count / n_cases) * 100:.1f}%"
        })
        
    output_blocks.append(
        OutputBlock(
            block_type=OutputBlockType.TABLE,
            title="Summary of Missing Values",
            content={
                "columns": ["Missing Values", "Number of Cases", "Percent of Total"],
                "rows": pattern_rows,
                "footnotes": []
            }
        )
    )

    return NormalizedResult(
        title="Missing Value Analysis",
        variables={"analyzed": variables},
        output_blocks=output_blocks,
        interpretation={"academic_sentence": "A missing value analysis was performed to identify patterns of missingness."}
    )
