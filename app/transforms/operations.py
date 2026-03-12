"""KALESS Engine — Transformation Operations.

Pandas-based implementations for data transformation.
Prioritizes strict schema fidelity and canonical dataset formats.
"""

from __future__ import annotations

import pandas as pd
import numpy as np

def compute_variable(df: pd.DataFrame, target_col: str, expression: str) -> pd.DataFrame:
    """Evaluate a mathematical expression into a new or existing column.
    Uses pandas.eval which supports basic math (+, -, *, /) and math functions.
    """
    try:
        df[target_col] = df.eval(expression)
    except Exception as e:
        raise ValueError(f"Failed to evaluate compute expression: {str(e)}")
    return df

def z_score(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Standardize (Z-score) the selected columns.
    Creates new columns named `z_{col}`.
    """
    for col in columns:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in dataset.")
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise ValueError(f"Column '{col}' must be numeric for z-score standardization.")
            
        mean = df[col].mean()
        std = df[col].std()
        
        if std == 0:
            df[f"z_{col}"] = 0.0
        else:
            df[f"z_{col}"] = (df[col] - mean) / std
            
    return df

def recode(df: pd.DataFrame, column: str, target_col: str, mapping: dict[str, str | float | int], default_value: Any = None) -> pd.DataFrame:
    """Recode values in a column to new values."""
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in dataset.")
        
    def _apply_mapping(val):
        str_val = str(val)
        if str_val in mapping:
            return mapping[str_val]
        if val in mapping:
            return mapping[val]
        return default_value if default_value is not None else val

    df[target_col] = df[column].apply(_apply_mapping)
    return df

def reverse_code(df: pd.DataFrame, columns: list[str], min_val: float, max_val: float) -> pd.DataFrame:
    """Reverse code a numeric scale. 
    Formula: max_val + min_val - x
    """
    for col in columns:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found.")
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise ValueError(f"Column '{col}' must be numeric for reverse coding.")
            
        new_col = f"rev_{col}"
        df[new_col] = (max_val + min_val) - df[col]
        
    return df

def filter_cases(df: pd.DataFrame, condition: str) -> pd.DataFrame:
    """Keep only rows matching the condition."""
    try:
        return df.query(condition).copy()
    except Exception as e:
        raise ValueError(f"Failed to apply filter condition: {str(e)}")

def sort_cases(df: pd.DataFrame, column: str, ascending: bool = True) -> pd.DataFrame:
    """Sort the dataset by a column."""
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found.")
    return df.sort_values(by=column, ascending=ascending).reset_index(drop=True)
