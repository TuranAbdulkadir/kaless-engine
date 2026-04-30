"""Survival Analysis (Kaplan-Meier)."""

import pandas as pd
import numpy as np
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from app.schemas.results import NormalizedResult, OutputBlock, ChartData, OutputBlockType

def run_survival_analysis(df: pd.DataFrame, time_var: str, status_var: str, factor_var: str = None) -> NormalizedResult:
    """Run Kaplan-Meier Survival Analysis."""
    df_clean = df.dropna(subset=[time_var, status_var] + ([factor_var] if factor_var else [])).copy()
    
    if len(df_clean) == 0:
        raise ValueError("No valid cases found after dropping missing values.")
        
    T = df_clean[time_var]
    E = df_clean[status_var]
    
    kmf = KaplanMeierFitter()
    
    output_blocks = []
    charts = []
    
    if not factor_var:
        kmf.fit(T, event_observed=E, label='All Cases')
        
        # Summary Table
        output_blocks.append(
            OutputBlock(
                block_type=OutputBlockType.TABLE,
                title="Survival Summary",
                content={
                    "columns": ["Median Survival Time"],
                    "rows": [{"Median Survival Time": str(kmf.median_survival_time_)}],
                    "footnotes": []
                }
            )
        )
        
        # Chart
        chart_data = []
        for t, s in zip(kmf.survival_function_.index, kmf.survival_function_['All Cases']):
            chart_data.append({"Time": float(t), "Survival": float(s)})
            
        charts.append(
            ChartData(
                chart_type="line",
                data=chart_data,
                config={"title": "Kaplan-Meier Survival Curve", "x_axis": time_var, "y_axis": "Cum Survival"}
            )
        )
    else:
        # Compare groups
        groups = df_clean[factor_var].unique()
        if len(groups) > 10:
            raise ValueError(f"Too many groups in factor '{factor_var}' (>10).")
            
        summary_rows = []
        for g in groups:
            mask = (df_clean[factor_var] == g)
            kmf.fit(T[mask], event_observed=E[mask], label=str(g))
            summary_rows.append({"Group": str(g), "Median Survival": str(kmf.median_survival_time_)})
            
        output_blocks.append(
            OutputBlock(
                block_type=OutputBlockType.TABLE,
                title="Means and Medians for Survival Time",
                content={
                    "columns": ["Group", "Median Survival"],
                    "rows": summary_rows,
                    "footnotes": []
                }
            )
        )
        
        if len(groups) == 2:
            mask0 = (df_clean[factor_var] == groups[0])
            mask1 = (df_clean[factor_var] == groups[1])
            res = logrank_test(T[mask0], T[mask1], event_observed_A=E[mask0], event_observed_B=E[mask1])
            output_blocks.append(
                OutputBlock(
                    block_type=OutputBlockType.TABLE,
                    title="Overall Comparisons",
                    content={
                        "columns": ["Test", "Chi-Square", "df", "Sig."],
                        "rows": [
                            {"Test": "Log Rank (Mantel-Cox)", "Chi-Square": f"{res.test_statistic:.3f}", "df": "1", "Sig.": f"{res.p_value:.3f}"}
                        ],
                        "footnotes": ["Test of equality of survival distributions for the different levels of factor."]
                    }
                )
            )

    return NormalizedResult(
        title="Kaplan-Meier Survival Analysis",
        variables={"analyzed": [time_var, status_var] + ([factor_var] if factor_var else [])},
        output_blocks=output_blocks,
        charts=charts,
        interpretation={"academic_sentence": "A Kaplan-Meier survival analysis was conducted."}
    )
