"""KALESS Engine — Transformation Operations.

Pandas-based implementations for data transformation.
Prioritizes strict schema fidelity and canonical dataset formats.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Any

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

def recode(df: pd.DataFrame, column: str, target_col: str, rules: list[dict], default_value: Any = None) -> pd.DataFrame:
    """Recode values in a column to new values using rules."""
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in dataset.")
        
    def _apply_mapping(val):
        for rule in rules:
            if rule.get("type") == "range":
                try:
                    num_val = float(val)
                    if rule["min"] <= num_val <= rule["max"]:
                        return rule["new_value"]
                except (ValueError, TypeError):
                    pass
            elif rule.get("type") == "value":
                if str(val) == str(rule["old_value"]) or val == rule["old_value"]:
                    return rule["new_value"]
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

def transpose_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Transpose the dataset. Rows become columns and columns become rows."""
    # Transposing can create huge datasets. We limit the number of rows/cols
    if df.shape[0] > 10000:
        raise ValueError("Dataset is too large to transpose safely (>10,000 rows).")
    return df.T

def rank_cases(df: pd.DataFrame, column: str, target_col: str, ascending: bool = True) -> pd.DataFrame:
    """Assign ranks to numeric values."""
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found.")
    df[target_col] = df[column].rank(ascending=ascending, method='average')
    return df

def count_values(df: pd.DataFrame, target_col: str, columns: list[str], value_to_count: Any) -> pd.DataFrame:
    """Count occurrences of a specific value across multiple columns within each case."""
    for col in columns:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found.")
    
    # Check if value_to_count is in each row for the specified columns and sum
    df[target_col] = (df[columns] == value_to_count).sum(axis=1)
    return df

def merge_datasets(df1: pd.DataFrame, df2: pd.DataFrame, merge_type: str, key_col: str = None) -> pd.DataFrame:
    """Merge two datasets.
    merge_type: 'add_cases' (concat rows) or 'add_variables' (merge on key or index).
    """
    if merge_type == 'add_cases':
        return pd.concat([df1, df2], ignore_index=True)
    elif merge_type == 'add_variables':
        if key_col:
            if key_col not in df1.columns or key_col not in df2.columns:
                raise ValueError(f"Key column '{key_col}' must exist in both datasets.")
            return pd.merge(df1, df2, on=key_col, how='outer')
        else:
            return pd.concat([df1, df2], axis=1)
    else:
        raise ValueError("Invalid merge type. Use 'add_cases' or 'add_variables'.")

def automatic_recode(df: pd.DataFrame, column: str, target_col: str) -> pd.DataFrame:
    """Automatically recode string values into numeric categories."""
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found.")
    
    # Use pandas factorize to assign numeric IDs (0-indexed, we'll do 1-indexed like SPSS)
    codes, uniques = pd.factorize(df[column])
    df[target_col] = codes + 1  # 1, 2, 3...
    
    # In a full implementation, we would also return the value labels metadata.
    # For now, the backend will just do the numeric transformation.
    return df

def visual_binning(df: pd.DataFrame, column: str, target_col: str, cutpoints: list[float], labels: list[str] = None) -> pd.DataFrame:
    """Bin continuous variables into categories based on cutpoints."""
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found.")
    if not pd.api.types.is_numeric_dtype(df[column]):
        raise ValueError(f"Column '{column}' must be numeric for visual binning.")
        
    # We need n cutpoints to create n+1 bins.
    # So we pad the cutpoints with -inf and inf.
    bins = [-np.inf] + sorted(cutpoints) + [np.inf]
    
    if labels and len(labels) == len(bins) - 1:
        df[target_col] = pd.cut(df[column], bins=bins, labels=labels, right=True)
    else:
        # Default labels: 1, 2, 3...
        df[target_col] = pd.cut(df[column], bins=bins, labels=False, right=True) + 1
        
    return df

def add_variable(df: pd.DataFrame, target_col: str, default_value: Any = None) -> pd.DataFrame:
    """Add a new variable (column) to the dataset, filled with a default value (e.g., None/NaN)."""
    if target_col in df.columns:
        raise ValueError(f"Column '{target_col}' already exists.")
    df[target_col] = default_value
    return df
