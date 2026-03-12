"""KALESS Engine — Dataset Parser Core.

Parses CSV, XLSX, and TSV files. Returns structured metadata,
inferred column types, preview rows, and import warnings.
"""

from __future__ import annotations

import io
import math
from typing import Any

import numpy as np
import pandas as pd

from app.utils.errors import ParseError


# Type inference mapping
def _infer_kaless_type(series: pd.Series) -> str:
    """Map a pandas series dtype to KALESS variable type."""
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "date"
    return "string"


def _infer_measure_level(series: pd.Series, col_type: str) -> str:
    """Infer measure level from data characteristics."""
    if col_type == "boolean":
        return "nominal"
    if col_type == "date":
        return "scale"
    if col_type == "string":
        n_unique = series.nunique()
        n_total = len(series)
        if n_unique <= 2:
            return "nominal"
        if n_unique <= 10 or (n_total > 0 and n_unique / n_total < 0.05):
            return "nominal"
        return "nominal"
    # numeric
    if col_type == "numeric":
        n_unique = series.nunique()
        if n_unique <= 2:
            return "nominal"
        if n_unique <= 7:
            return "ordinal"
        return "scale"
    return "scale"


def _safe_value(v: Any) -> Any:
    """Convert numpy/pandas types to JSON-safe Python types."""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, pd.Timestamp):
        return v.isoformat()
    return v


def parse_dataset(
    file_content: bytes,
    file_type: str,
    encoding: str = "utf-8",
    delimiter: str | None = None,
    max_preview_rows: int = 100,
) -> dict[str, Any]:
    """Parse a dataset file and return structured metadata.

    Args:
        file_content: Raw file bytes
        file_type: One of 'csv', 'xlsx', 'tsv'
        encoding: Character encoding (for CSV/TSV)
        delimiter: Delimiter override (for CSV)
        max_preview_rows: Number of preview rows to return

    Returns:
        Structured parse result with schema, preview, and warnings.
    """
    warnings: list[str] = []

    try:
        df = _read_file(file_content, file_type, encoding, delimiter, warnings)
    except Exception as e:
        raise ParseError(f"Failed to read file: {str(e)}")

    if df.empty:
        raise ParseError("File is empty or contains no data rows.")

    # Clean column names
    original_cols = list(df.columns)
    df.columns = [_clean_column_name(str(c), i) for i, c in enumerate(df.columns)]
    renamed = [(orig, new) for orig, new in zip(original_cols, df.columns) if str(orig) != new]
    if renamed:
        warnings.append(
            f"{len(renamed)} column(s) were renamed to valid identifiers."
        )

    # Check for duplicate columns
    if df.columns.duplicated().any():
        dupes = df.columns[df.columns.duplicated()].unique().tolist()
        warnings.append(f"Duplicate column names detected and auto-suffixed: {dupes}")
        df.columns = pd.io.common.dedup_names(list(df.columns), is_potential_multiindex=False)

    row_count = len(df)
    col_count = len(df.columns)

    # Infer column metadata
    columns: list[dict[str, Any]] = []
    for i, col in enumerate(df.columns):
        series = df[col]
        col_type = _infer_kaless_type(series)
        missing_count = int(series.isna().sum())
        n_unique = int(series.nunique())

        # Detect mixed types
        if col_type == "string" and series.dropna().apply(type).nunique() > 1:
            warnings.append(f"Column '{col}' has mixed data types.")

        # Detect mostly-missing columns
        if row_count > 0 and missing_count / row_count > 0.5:
            warnings.append(
                f"Column '{col}' is more than 50% missing ({missing_count}/{row_count})."
            )

        columns.append({
            "index": i,
            "name": str(col),
            "inferred_type": col_type,
            "measure_level": _infer_measure_level(series, col_type),
            "missing_count": missing_count,
            "unique_count": n_unique,
            "sample_values": [
                _safe_value(v) for v in series.dropna().head(5).tolist()
            ],
        })

    # Build preview rows
    preview_df = df.head(max_preview_rows)
    preview_rows: list[dict[str, Any]] = []
    for _, row in preview_df.iterrows():
        preview_rows.append({
            str(col): _safe_value(row[col]) for col in df.columns
        })

    return {
        "row_count": row_count,
        "column_count": col_count,
        "columns": columns,
        "preview_rows": preview_rows,
        "warnings": warnings,
        "encoding": encoding,
        "delimiter": delimiter,
        "_df": df,  # Passed internally for saving to canonical format
    }


def _read_file(
    content: bytes,
    file_type: str,
    encoding: str,
    delimiter: str | None,
    warnings: list[str],
) -> pd.DataFrame:
    """Read file bytes into a DataFrame."""
    if file_type == "csv":
        sep = delimiter or ","
        try:
            df = pd.read_csv(io.BytesIO(content), encoding=encoding, sep=sep)
        except UnicodeDecodeError:
            warnings.append(
                f"Encoding '{encoding}' failed. Retrying with 'latin-1'."
            )
            df = pd.read_csv(io.BytesIO(content), encoding="latin-1", sep=sep)
        return df

    elif file_type == "tsv":
        try:
            df = pd.read_csv(io.BytesIO(content), encoding=encoding, sep="\t")
        except UnicodeDecodeError:
            warnings.append(
                f"Encoding '{encoding}' failed. Retrying with 'latin-1'."
            )
            df = pd.read_csv(io.BytesIO(content), encoding="latin-1", sep="\t")
        return df

    elif file_type == "xlsx":
        df = pd.read_excel(io.BytesIO(content), engine="openpyxl")
        return df

    elif file_type == "parquet":
        df = pd.read_parquet(io.BytesIO(content))
        return df

    else:
        raise ParseError(f"Unsupported file type: {file_type}")


def _clean_column_name(name: str, index: int) -> str:
    """Clean a column name to be a valid identifier."""
    name = name.strip()
    if not name or name.startswith("Unnamed"):
        return f"var_{index + 1}"
    # Replace spaces and special chars
    cleaned = ""
    for ch in name:
        if ch.isalnum() or ch == "_":
            cleaned += ch
        elif ch in (" ", "-", "."):
            cleaned += "_"
    # Ensure starts with letter or underscore
    if cleaned and cleaned[0].isdigit():
        cleaned = "v_" + cleaned
    return cleaned or f"var_{index + 1}"
