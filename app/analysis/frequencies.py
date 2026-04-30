"""KALESS Engine — Frequencies Module."""

import time
from datetime import datetime
import pandas as pd

from app.core.preprocessing import validate_variable_exists
from app.schemas.results import NormalizedResult, ChartData, GroupDescriptive

def run_frequencies(df: pd.DataFrame, variable: str) -> NormalizedResult:
    start = time.time()
    warnings: list[str] = []

    try:
        validate_variable_exists(df, variable)
        
        series = df[variable]
        n_missing = int(series.isna().sum())
        if n_missing > 0:
            warnings.append(f"{n_missing} missing value(s) excluded.")
            
        series = series.dropna()
        freq = series.value_counts().sort_index()
        total = len(series)

        freq_table = []
        cumulative = 0
        for val, count in freq.items():
            cumulative += count
            freq_table.append({
                "value": str(val),
                "frequency": int(count),
                "percent": round(count / total * 100, 1) if total > 0 else 0,
                "cumulative_percent": round(cumulative / total * 100, 1) if total > 0 else 0,
            })

        chart_data = [{"category": str(val), "count": int(count)} for val, count in freq.items()]
        
        # Decide chart type
        chart_type = "bar" if len(freq) > 10 else "pie"

        duration = int((time.time() - start) * 1000)

        return NormalizedResult(
            analysis_type="frequencies",
            title=f"Frequencies: {variable}",
            variables={"analyzed": [variable]},
            frequency_table=freq_table,
            descriptives=[GroupDescriptive(name=variable, n=total)],
            charts=[ChartData(
                chart_type=chart_type,
                data=chart_data,
                config={"title": f"Frequencies of {variable}", "x_label": variable, "y_label": "Count"},
            )],
            warnings=warnings,
            metadata={
                "n_total": len(df),
                "missing_excluded": n_missing,
                "library": "pandas",
                "duration_ms": duration,
                "timestamp": datetime.utcnow().isoformat(),
            },
        )
    except Exception as e:
        raise ValueError(f"Frequencies computation failed: {str(e)}")
