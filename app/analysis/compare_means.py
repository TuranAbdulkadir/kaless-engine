"""KALESS Engine — Compare Means Module.

Implements the 'Means' procedure (descriptive statistics broken down by layers/groups).
"""

from __future__ import annotations

import time
from datetime import datetime

import pandas as pd

from app.core.preprocessing import validate_variable_exists
from app.schemas.results import NormalizedResult, OutputBlock, OutputBlockType


def run_means(
    df: pd.DataFrame,
    dependent: list[str],
    independent: list[str]
) -> NormalizedResult:
    """Run Compare Means (Means procedure).
    
    Generates Report table: Mean, N, Std. Deviation for dependent variables
    grouped by independent variables.
    """
    start = time.time()
    warnings: list[str] = []

    if not dependent or not independent:
        raise ValueError("Compare Means requires at least one dependent and one independent variable.")

    valid_dep = [v for v in dependent if v in df.columns]
    valid_indep = [v for v in independent if v in df.columns]

    if not valid_dep or not valid_indep:
        raise ValueError("Provided variables not found in dataset.")

    df_clean = df[valid_dep + valid_indep].dropna()
    n_excluded = len(df) - len(df_clean)
    if n_excluded > 0:
        warnings.append(f"{n_excluded} case(s) excluded due to missing values.")

    output_blocks = []

    # Means Report Table
    # For each dependent variable, group by independent variables
    for dep in valid_dep:
        summary = df_clean.groupby(valid_indep)[dep].agg(['mean', 'count', 'std']).round(3)
        summary.rename(columns={"mean": "Mean", "count": "N", "std": "Std. Deviation"}, inplace=True)
        
        rows = []
        for idx, data in summary.iterrows():
            idx_tuple = (idx,) if not isinstance(idx, tuple) else idx
            row_data = {valid_indep[i]: str(idx_tuple[i]) for i in range(len(valid_indep))}
            row_data["Mean"] = data["Mean"]
            row_data["N"] = int(data["N"])
            row_data["Std. Deviation"] = data["Std. Deviation"]
            rows.append(row_data)

        # Total row
        total_mean = round(df_clean[dep].mean(), 3)
        total_n = int(df_clean[dep].count())
        total_std = round(df_clean[dep].std(), 3)
        
        total_row = {valid_indep[0]: "Total", "Mean": total_mean, "N": total_n, "Std. Deviation": total_std}
        for indep in valid_indep[1:]:
            total_row[indep] = ""
        rows.append(total_row)

        columns = valid_indep + ["Mean", "N", "Std. Deviation"]

        output_blocks.append(
            OutputBlock(
                block_type=OutputBlockType.TABLE,
                title=f"Report: {dep}",
                content={
                    "columns": columns,
                    "rows": rows
                }
            )
        )

    duration = int((time.time() - start) * 1000)

    return NormalizedResult(
        analysis_type="means",
        title="Compare Means",
        variables={"dependent": dependent, "independent": independent},
        output_blocks=output_blocks,
        warnings=warnings,
        metadata={
            "n_total": len(df),
            "missing_excluded": n_excluded,
            "library": "pandas",
            "duration_ms": duration,
            "timestamp": datetime.utcnow().isoformat(),
        }
    )
