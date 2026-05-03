"""KALESS Engine — Reliability Analysis Module (Cronbach's Alpha).

Implements the full SPSS Reliability Analysis workflow:
  1. Case Processing Summary
  2. Reliability Statistics (Cronbach's Alpha)
  3. Item-Total Statistics (Scale if Item Deleted)

Uses the raw Cronbach's Alpha formula with pandas/numpy — no external library shortcut.
"""

import time
from datetime import datetime

import numpy as np
import pandas as pd

from app.core.preprocessing import validate_variable_exists, validate_numeric
from app.schemas.results import NormalizedResult, OutputBlock, OutputBlockType


def _cronbach_alpha(data: pd.DataFrame) -> float:
    """Calculate Cronbach's Alpha using the raw variance formula.

    Formula: alpha = (k / (k - 1)) * (1 - sum(item_variances) / total_variance)
    """
    k = data.shape[1]
    if k < 2:
        return 0.0
    item_variances = data.var(axis=0, ddof=1)
    total_scores = data.sum(axis=1)
    total_variance = total_scores.var(ddof=1)
    if total_variance == 0:
        return 0.0
    return float((k / (k - 1)) * (1 - item_variances.sum() / total_variance))


def run_reliability(df: pd.DataFrame, variables: list[str], item_deleted: bool = True) -> NormalizedResult:
    """Run Reliability Analysis producing SPSS-style output tables.

    Args:
        df: The dataset.
        variables: List of item/scale variable names (minimum 2).
        item_deleted: If True, compute the Item-Total Statistics table.

    Returns:
        NormalizedResult with Case Processing Summary, Reliability Statistics,
        and (optionally) Item-Total Statistics.
    """
    start = time.time()
    warnings: list[str] = []

    # Validate all variables
    for var in variables:
        validate_variable_exists(df, var)
        validate_numeric(df[var], var)

    # Listwise deletion
    sub_df = df[variables].dropna()
    valid_n = len(sub_df)
    excluded_n = len(df) - valid_n

    if valid_n < 2:
        raise ValueError("Not enough valid cases for reliability analysis.")

    if excluded_n > 0:
        warnings.append(f"{excluded_n} case(s) excluded listwise.")

    k = len(variables)

    # ═══ Cronbach's Alpha (main) ═══
    alpha = _cronbach_alpha(sub_df)

    output_blocks: list[OutputBlock] = []

    # ── TABLE 1: Case Processing Summary ──
    total_n = valid_n + excluded_n
    output_blocks.append(OutputBlock(
        block_type=OutputBlockType.TABLE,
        title="Case Processing Summary",
        display_order=1,
        content={
            "columns": ["", "N", "%"],
            "rows": [
                {"": "Valid", "N": str(valid_n), "%": f"{(valid_n / total_n * 100):.1f}"},
                {"": "Excluded(a)", "N": str(excluded_n), "%": f"{(excluded_n / total_n * 100):.1f}"},
                {"": "Total", "N": str(total_n), "%": "100.0"},
            ],
            "footnotes": ["a. Listwise deletion based on all variables in the procedure."],
        },
    ))

    # ── TABLE 2: Reliability Statistics ──
    output_blocks.append(OutputBlock(
        block_type=OutputBlockType.TABLE,
        title="Reliability Statistics",
        display_order=2,
        content={
            "columns": ["Cronbach's Alpha", "N of Items"],
            "rows": [{"Cronbach's Alpha": f"{alpha:.3f}", "N of Items": str(k)}],
        },
    ))

    # ── TABLE 3: Item-Total Statistics ──
    if item_deleted and k >= 3:
        item_total_rows = []
        total_scores = sub_df.sum(axis=1)

        for var in variables:
            # Scale if item deleted
            remaining = [v for v in variables if v != var]
            remaining_df = sub_df[remaining]
            remaining_scores = remaining_df.sum(axis=1)

            scale_mean_deleted = float(remaining_scores.mean())
            scale_var_deleted = float(remaining_scores.var(ddof=1))

            # Corrected Item-Total Correlation:
            # Pearson r between the item score and the sum of the remaining items
            item_scores = sub_df[var]
            corrected_total = total_scores - item_scores
            if corrected_total.std() == 0 or item_scores.std() == 0:
                corrected_itc = 0.0
            else:
                corrected_itc = float(np.corrcoef(item_scores, corrected_total)[0, 1])

            # Alpha if item deleted
            alpha_deleted = _cronbach_alpha(remaining_df)

            item_total_rows.append({
                "": var,
                "Scale Mean if Item Deleted": f"{scale_mean_deleted:.4f}",
                "Scale Variance if Item Deleted": f"{scale_var_deleted:.4f}",
                "Corrected Item-Total Correlation": f"{corrected_itc:.4f}",
                "Cronbach's Alpha if Item Deleted": f"{alpha_deleted:.3f}",
            })

        output_blocks.append(OutputBlock(
            block_type=OutputBlockType.TABLE,
            title="Item-Total Statistics",
            display_order=3,
            content={
                "columns": [
                    "",
                    "Scale Mean if Item Deleted",
                    "Scale Variance if Item Deleted",
                    "Corrected Item-Total Correlation",
                    "Cronbach's Alpha if Item Deleted",
                ],
                "rows": item_total_rows,
            },
        ))

    duration = int((time.time() - start) * 1000)

    from app.utils.interpretation import generate_interpretation

    res = NormalizedResult(
        analysis_type="reliability",
        title="Reliability Analysis",
        variables={"items": variables},
        output_blocks=output_blocks,
        warnings=warnings,
        metadata={
            "cronbach_alpha": round(alpha, 4),
            "n_items": k,
            "valid_n": valid_n,
            "excluded_n": excluded_n,
            "model": "Alpha (Cronbach)",
            "library": "numpy/pandas (raw formula)",
            "duration_ms": duration,
            "timestamp": datetime.utcnow().isoformat(),
        },
    )
    res.interpretation = generate_interpretation(res)
    return res
