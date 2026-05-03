"""KALESS Engine — T-Test Module."""

import time
from datetime import datetime
import pandas as pd
from scipy import stats

from app.core.preprocessing import validate_variable_exists, validate_numeric, drop_missing_listwise
from app.schemas.results import NormalizedResult, PrimaryResult, GroupDescriptive, SignificanceLevel, OutputBlock, OutputBlockType
from app.utils.interpretation import generate_interpretation

def get_significance_level(p: float, alpha: float = 0.05) -> SignificanceLevel:
    if p < alpha:
        return SignificanceLevel.SIGNIFICANT
    return SignificanceLevel.NOT_SIGNIFICANT

def calculate_independent_t(df: pd.DataFrame, test_var: str, group_var: str, group1_val, group2_val, alpha: float = 0.05) -> NormalizedResult:
    start = time.time()
    warnings = []
    
    try:
        import numpy as np
        validate_variable_exists(df, test_var)
        validate_variable_exists(df, group_var)
        validate_numeric(df[test_var], test_var)
        
        # Cast group_var values to string if needed, or parse group1_val/group2_val to match df types
        try:
            # Attempt to infer types
            if pd.api.types.is_numeric_dtype(df[group_var]):
                group1_val = float(group1_val)
                group2_val = float(group2_val)
        except:
            pass

        cleaned, n_dropped = drop_missing_listwise(df, [test_var, group_var])
        if n_dropped > 0:
            warnings.append(f"{n_dropped} missing cases excluded.")
            
        g1_data = cleaned[cleaned[group_var] == group1_val][test_var]
        g2_data = cleaned[cleaned[group_var] == group2_val][test_var]
        
        if len(g1_data) < 2 or len(g2_data) < 2:
            raise ValueError(f"One or both groups ({group1_val}, {group2_val}) have insufficient valid cases (N<2).")
            
        # Levene's Test
        levene_stat, levene_p = stats.levene(g1_data, g2_data)
        
        # Means and Diff
        n1, n2 = len(g1_data), len(g2_data)
        m1, m2 = g1_data.mean(), g2_data.mean()
        v1, v2 = g1_data.var(ddof=1), g2_data.var(ddof=1)
        mean_diff = m1 - m2
        
        # Equal variances assumed
        t_stat_eq, p_val_eq = stats.ttest_ind(g1_data, g2_data, equal_var=True)
        df_eq = n1 + n2 - 2
        sp = np.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / df_eq)
        se_diff_eq = sp * np.sqrt(1 / n1 + 1 / n2)
        ci_eq_margin = stats.t.ppf(1 - alpha/2, df_eq) * se_diff_eq
        
        # Equal variances not assumed (Welch)
        t_stat_uneq, p_val_uneq = stats.ttest_ind(g1_data, g2_data, equal_var=False)
        se_diff_uneq = np.sqrt(v1/n1 + v2/n2)
        df_uneq = (v1/n1 + v2/n2)**2 / ((v1/n1)**2 / (n1 - 1) + (v2/n2)**2 / (n2 - 1))
        ci_uneq_margin = stats.t.ppf(1 - alpha/2, df_uneq) * se_diff_uneq
        
        descriptives = [
            GroupDescriptive(name=f"{group1_val}", n=n1, mean=float(m1), sd=float(np.sqrt(v1)), se=float(np.sqrt(v1)/np.sqrt(n1))),
            GroupDescriptive(name=f"{group2_val}", n=n2, mean=float(m2), sd=float(np.sqrt(v2)), se=float(np.sqrt(v2)/np.sqrt(n2))),
        ]
        
        # For the primary result, we default to the equal variances assumed (classic)
        primary = PrimaryResult(
            statistic_name="t",
            statistic_value=float(t_stat_eq),
            df=float(df_eq),
            p_value=float(p_val_eq),
            p_value_formatted=f"p < .001" if p_val_eq < 0.001 else f"p = {p_val_eq:.3f}",
            significance=get_significance_level(p_val_eq, alpha),
            alpha=alpha
        )

        output_blocks = [
            OutputBlock(
                block_type=OutputBlockType.TABLE,
                title="Group Statistics",
                content={
                    "columns": [group_var, "N", "Mean", "Std. Deviation", "Std. Error Mean"],
                    "rows": [
                        {
                            group_var: d.name,
                            "N": d.n,
                            "Mean": f"{d.mean:.3f}",
                            "Std. Deviation": f"{d.sd:.3f}",
                            "Std. Error Mean": f"{d.se:.3f}"
                        } for d in descriptives
                    ]
                }
            ),
            OutputBlock(
                block_type=OutputBlockType.TABLE,
                title="Independent Samples Test",
                content={
                    "columns": ["Variances", "Levene F", "Levene Sig.", "t", "df", "Sig. (2-tailed)", "Mean Difference", "Std. Error Difference", "95% CI Lower", "95% CI Upper"],
                    "rows": [
                        {
                            "Variances": "Equal variances assumed",
                            "Levene F": f"{levene_stat:.3f}",
                            "Levene Sig.": f"{levene_p:.3f}",
                            "t": f"{t_stat_eq:.3f}",
                            "df": f"{df_eq}",
                            "Sig. (2-tailed)": f"{p_val_eq:.3f}",
                            "Mean Difference": f"{mean_diff:.5f}",
                            "Std. Error Difference": f"{se_diff_eq:.5f}",
                            "95% CI Lower": f"{(mean_diff - ci_eq_margin):.4f}",
                            "95% CI Upper": f"{(mean_diff + ci_eq_margin):.4f}"
                        },
                        {
                            "Variances": "Equal variances not assumed",
                            "Levene F": "",
                            "Levene Sig.": "",
                            "t": f"{t_stat_uneq:.3f}",
                            "df": f"{df_uneq:.3f}",
                            "Sig. (2-tailed)": f"{p_val_uneq:.3f}",
                            "Mean Difference": f"{mean_diff:.5f}",
                            "Std. Error Difference": f"{se_diff_uneq:.5f}",
                            "95% CI Lower": f"{(mean_diff - ci_uneq_margin):.4f}",
                            "95% CI Upper": f"{(mean_diff + ci_uneq_margin):.4f}"
                        }
                    ]
                }
            )
        ]
        
        res = NormalizedResult(
            analysis_type="independent_t",
            title="Independent Samples T-Test",
            variables={"test_variable": test_var, "grouping_variable": group_var},
            descriptives=descriptives,
            primary=primary,
            output_blocks=output_blocks,
            warnings=warnings,
            metadata={"n_total": len(df), "valid_n": len(cleaned), "duration_ms": int((time.time() - start) * 1000), "timestamp": datetime.utcnow().isoformat()}
        )
        res.interpretation = generate_interpretation(res)
        return res
    except Exception as e:
        raise ValueError(f"Independent T-Test failed: {str(e)}")

def calculate_paired_t(df: pd.DataFrame, var1: str, var2: str, alpha: float = 0.05) -> NormalizedResult:
    start = time.time()
    warnings = []
    
    try:
        validate_variable_exists(df, var1)
        validate_variable_exists(df, var2)
        validate_numeric(df[var1], var1)
        validate_numeric(df[var2], var2)
        
        cleaned, n_dropped = drop_missing_listwise(df, [var1, var2])
        if n_dropped > 0:
            warnings.append(f"{n_dropped} missing cases excluded.")
            
        v1_data, v2_data = cleaned[var1], cleaned[var2]
        
        if len(v1_data) == 0:
            raise ValueError("Zero valid cases after listwise deletion.")
            
        t_stat, p_val = stats.ttest_rel(v1_data, v2_data)
        
        m1, m2 = v1_data.mean(), v2_data.mean()
        mean_diff = m1 - m2
        df_val = len(v1_data) - 1
        
        descriptives = [
            GroupDescriptive(name=var1, n=len(v1_data), mean=float(m1), sd=float(v1_data.std()), se=float(v1_data.sem())),
            GroupDescriptive(name=var2, n=len(v2_data), mean=float(m2), sd=float(v2_data.std()), se=float(v2_data.sem())),
        ]
        
        primary = PrimaryResult(
            statistic_name="t",
            statistic_value=float(t_stat),
            df=float(df_val),
            p_value=float(p_val),
            p_value_formatted=f"p < .001" if p_val < 0.001 else f"p = {p_val:.3f}",
            significance=get_significance_level(p_val, alpha),
            alpha=alpha
        )

        output_blocks = [
            OutputBlock(
                block_type=OutputBlockType.TABLE,
                title="Paired Samples Test",
                content={
                    "columns": ["Pair", "Mean Difference", "t", "df", "Sig. (2-tailed)"],
                    "rows": [{
                        "Pair": f"{var1} - {var2}",
                        "Mean Difference": f"{mean_diff:.3f}",
                        "t": f"{t_stat:.3f}",
                        "df": f"{df_val}",
                        "Sig. (2-tailed)": f"{p_val:.3f}"
                    }]
                }
            )
        ]
        
        res = NormalizedResult(
            analysis_type="paired_t",
            title="Paired Samples T-Test",
            variables={"pair": [var1, var2]},
            descriptives=descriptives,
            primary=primary,
            output_blocks=output_blocks,
            warnings=warnings,
            metadata={"n_total": len(df), "valid_n": len(cleaned), "duration_ms": int((time.time() - start) * 1000), "timestamp": datetime.utcnow().isoformat()}
        )
        res.interpretation = generate_interpretation(res)
        return res
    except Exception as e:
        raise ValueError(f"Paired T-Test failed: {str(e)}")
def run_one_sample_t_test(df: pd.DataFrame, variable: str, test_value: float = 0.0, alpha: float = 0.05) -> NormalizedResult:
    start = time.time()
    warnings = []

    validate_variable_exists(df, variable)
    validate_numeric(df[variable], variable)

    cleaned = df[variable].dropna()
    valid_n = len(cleaned)

    if valid_n < 2:
        raise ValueError("Not enough valid cases for One-Sample T-Test.")

    if len(cleaned) < len(df):
        warnings.append(f"{len(df) - valid_n} cases excluded.")

    t_stat, p_val = stats.ttest_1samp(cleaned, test_value)
    mean = float(cleaned.mean())
    std = float(cleaned.std())
    mean_diff = mean - test_value
    df_val = valid_n - 1

    one_sample_stats = [{
        "N": valid_n,
        "Mean": round(mean, 4),
        "Std. Deviation": round(std, 4),
        "Std. Error Mean": round(float(cleaned.sem()), 4)
    }]

    one_sample_test = [{
        "Variable": variable,
        "t": round(t_stat, 3),
        "df": df_val,
        "Sig. (2-tailed)": round(p_val, 3),
        "Mean Difference": round(mean_diff, 4)
    }]

    output_blocks = [
        {
            "block_type": "table",
            "title": "One-Sample Statistics",
            "content": {
                "columns": ["N", "Mean", "Std. Deviation", "Std. Error Mean"],
                "rows": one_sample_stats
            }
        },
        {
            "block_type": "table",
            "title": "One-Sample Test",
            "content": {
                "columns": ["Variable", "t", "df", "Sig. (2-tailed)", "Mean Difference"],
                "rows": one_sample_test,
                "footnotes": [f"Test Value = {test_value}"]
            }
        }
    ]

    duration = int((time.time() - start) * 1000)
    res = NormalizedResult(
        analysis_type="one_sample_t_test",
        title="One-Sample T Test",
        variables={"test": [variable]},
        descriptives=[GroupDescriptive(name=variable, n=valid_n, mean=mean, sd=std)],
        output_blocks=output_blocks,
        warnings=warnings,
        primary=PrimaryResult(
            statistic_name="t",
            statistic_value=float(t_stat),
            df=float(df_val),
            p_value=float(p_val),
            p_value_formatted=f"p < .001" if p_val < 0.001 else f"p = {p_val:.3f}",
            significance=get_significance_level(p_val, alpha),
            alpha=alpha
        ),
        metadata={
            "valid_n": valid_n,
            "test_value": test_value,
            "duration_ms": duration,
            "timestamp": datetime.utcnow().isoformat()
        }
    )
    res.interpretation = generate_interpretation(res)
    return res
