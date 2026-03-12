"""KALESS Engine — Data Preprocessing Pipeline.

Validates variables, handles missing data, and prepares DataFrames
for statistical analysis.
"""

from __future__ import annotations

import pandas as pd
import numpy as np

from app.utils.errors import ValidationError


def validate_variable_exists(df: pd.DataFrame, var_name: str) -> None:
    """Ensure a variable exists in the DataFrame."""
    if var_name not in df.columns:
        raise ValidationError(f"Variable '{var_name}' not found in dataset.")


def validate_numeric(series: pd.Series, var_name: str) -> None:
    """Ensure a variable is numeric (or coercible)."""
    if not pd.api.types.is_numeric_dtype(series):
        raise ValidationError(
            f"Variable '{var_name}' must be numeric for this analysis."
        )


def validate_categorical(series: pd.Series, var_name: str) -> None:
    """Ensure a variable is categorical (string or low-cardinality)."""
    if pd.api.types.is_numeric_dtype(series):
        n_unique = series.nunique()
        if n_unique > 50:
            raise ValidationError(
                f"Variable '{var_name}' appears to be continuous (> 50 unique values), "
                "but a categorical variable is required."
            )


def validate_min_n(n: int, min_required: int, context: str = "") -> None:
    """Ensure minimum sample size."""
    if n < min_required:
        msg = f"Insufficient sample size: n={n}, minimum required is {min_required}."
        if context:
            msg += f" ({context})"
        raise ValidationError(msg)


def validate_exact_groups(
    series: pd.Series, var_name: str, expected: int
) -> list[str]:
    """Ensure a grouping variable has exactly N groups."""
    groups = sorted(series.dropna().unique().tolist())
    if len(groups) != expected:
        raise ValidationError(
            f"Variable '{var_name}' has {len(groups)} groups, "
            f"but exactly {expected} are required."
        )
    return [str(g) for g in groups]


def validate_min_groups(
    series: pd.Series, var_name: str, min_groups: int
) -> list[str]:
    """Ensure a grouping variable has at least N groups."""
    groups = sorted(series.dropna().unique().tolist())
    if len(groups) < min_groups:
        raise ValidationError(
            f"Variable '{var_name}' has {len(groups)} group(s), "
            f"but at least {min_groups} are required."
        )
    return [str(g) for g in groups]


def drop_missing_listwise(
    df: pd.DataFrame, columns: list[str]
) -> tuple[pd.DataFrame, int]:
    """Drop rows with any missing values in the specified columns.

    Returns:
        (cleaned_df, n_dropped)
    """
    before = len(df)
    cleaned = df[columns].dropna()
    dropped = before - len(cleaned)
    return cleaned, dropped


def coerce_numeric(
    series: pd.Series, var_name: str
) -> tuple[pd.Series, list[str]]:
    """Attempt to coerce a series to numeric, reporting warnings."""
    warnings: list[str] = []
    if pd.api.types.is_numeric_dtype(series):
        return series, warnings

    coerced = pd.to_numeric(series, errors="coerce")
    n_failed = int(coerced.isna().sum() - series.isna().sum())
    if n_failed > 0:
        warnings.append(
            f"{n_failed} non-numeric value(s) in '{var_name}' were converted to missing."
        )
    return coerced, warnings


def get_group_data(
    df: pd.DataFrame,
    dependent_var: str,
    grouping_var: str,
) -> dict[str, pd.Series]:
    """Split a dependent variable by groups.

    Returns dict of group_name -> Series (missing dropped).
    """
    result: dict[str, pd.Series] = {}
    for group_val, group_df in df.groupby(grouping_var):
        series = group_df[dependent_var].dropna()
        result[str(group_val)] = series
    return result


def compute_descriptive(series: pd.Series, name: str = "Overall") -> dict:
    """Compute descriptive statistics for a numeric series."""
    n = int(series.count())
    if n == 0:
        return {
            "name": name, "n": 0,
            "mean": None, "median": None, "sd": None, "se": None,
            "min": None, "max": None, "skewness": None, "kurtosis": None,
        }
    return {
        "name": name,
        "n": n,
        "mean": float(series.mean()),
        "median": float(series.median()),
        "sd": float(series.std(ddof=1)) if n > 1 else None,
        "se": float(series.sem()) if n > 1 else None,
        "min": float(series.min()),
        "max": float(series.max()),
        "skewness": float(series.skew()) if n > 2 else None,
        "kurtosis": float(series.kurtosis()) if n > 3 else None,
    }
