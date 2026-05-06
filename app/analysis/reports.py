"""KALESS Engine — Reports and Tables Module.

Implements Codebook, Case Summaries, OLAP Cubes, and Custom Tables.
"""

from __future__ import annotations

import time
from datetime import datetime

import pandas as pd

from app.schemas.results import NormalizedResult, OutputBlock, OutputBlockType

def run_codebook(df: pd.DataFrame, variables: list[str]) -> NormalizedResult:
    """Generate a codebook for variables."""
    start = time.time()
    
    if not variables:
        variables = df.columns.tolist()
        
    blocks = []
    
    for var in variables:
        if var not in df.columns:
            continue
            
        series = df[var]
        
        info = {
            "Variable Name": var,
            "Type": str(series.dtype),
            "N (Valid)": int(series.count()),
            "N (Missing)": int(series.isna().sum())
        }
        
        if pd.api.types.is_numeric_dtype(series):
            info["Mean"] = f"{series.mean():.3f}"
            info["Std. Dev"] = f"{series.std():.3f}"
            info["Min"] = f"{series.min():.3f}"
            info["Max"] = f"{series.max():.3f}"
        else:
            info["Unique Values"] = int(series.nunique())
            info["Top Value"] = str(series.mode().iloc[0]) if not series.mode().empty else "N/A"
            
        blocks.append(OutputBlock(
            block_type=OutputBlockType.TABLE,
            title=f"Codebook: {var}",
            content={
                "columns": ["Property", "Value"],
                "rows": [{"Property": k, "Value": v} for k, v in info.items()]
            }
        ))
        
    return NormalizedResult(
        analysis_type="codebook",
        title="Codebook",
        variables={"analyzed": variables},
        output_blocks=blocks,
        metadata={
            "n_total": len(df),
            "library": "pandas",
            "duration_ms": int((time.time() - start) * 1000),
            "timestamp": datetime.utcnow().isoformat(),
        }
    )

def run_case_summaries(df: pd.DataFrame, variables: list[str], grouping: str = None) -> NormalizedResult:
    """Generate Case Summaries."""
    start = time.time()
    
    if not variables:
        variables = df.columns.tolist()[:5] # Default to first 5
        
    valid_vars = [v for v in variables if v in df.columns]
    
    if grouping and grouping in df.columns:
        # Grouped case summaries
        summary = df.groupby(grouping)[valid_vars].describe().round(3)
        # Flatten multiindex for display
        rows = []
        for group_val, data in summary.iterrows():
            row_data = {"Group": str(group_val)}
            for col in summary.columns:
                row_data[f"{col[0]} ({col[1]})"] = data[col]
            rows.append(row_data)
            
        columns = ["Group"] + [f"{col[0]} ({col[1]})" for col in summary.columns]
        
    else:
        # Simple case summaries
        summary = df[valid_vars].describe().round(3).reset_index()
        summary.rename(columns={"index": "Statistic"}, inplace=True)
        columns = list(summary.columns)
        rows = summary.to_dict(orient="records")

    blocks = [
        OutputBlock(
            block_type=OutputBlockType.TABLE,
            title="Case Summaries",
            content={
                "columns": columns,
                "rows": rows
            }
        )
    ]
    
    return NormalizedResult(
        analysis_type="case_summaries",
        title="Case Summaries",
        variables={"analyzed": variables},
        output_blocks=blocks,
        metadata={
            "n_total": len(df),
            "library": "pandas",
            "duration_ms": int((time.time() - start) * 1000),
            "timestamp": datetime.utcnow().isoformat(),
        }
    )

def run_olap_cubes(df: pd.DataFrame, summary_vars: list[str], grouping_vars: list[str]) -> NormalizedResult:
    """Generate OLAP Cubes."""
    start = time.time()
    
    if not summary_vars or not grouping_vars:
        raise ValueError("OLAP Cubes requires at least one summary variable and one grouping variable.")
        
    valid_summary = [v for v in summary_vars if v in df.columns]
    valid_grouping = [v for v in grouping_vars if v in df.columns]
    
    if not valid_summary or not valid_grouping:
        raise ValueError("Variables not found in dataset.")
        
    grouped = df.groupby(valid_grouping)[valid_summary].agg(['sum', 'mean', 'count', 'std']).round(3)
    
    rows = []
    for idx, data in grouped.iterrows():
        # Handle single vs multiple grouping vars
        idx_tuple = (idx,) if not isinstance(idx, tuple) else idx
        
        row_data = {valid_grouping[i]: str(idx_tuple[i]) for i in range(len(valid_grouping))}
        
        for col in grouped.columns:
            row_data[f"{col[0]} ({col[1]})"] = data[col]
        rows.append(row_data)
        
    columns = valid_grouping + [f"{col[0]} ({col[1]})" for col in grouped.columns]
    
    blocks = [
        OutputBlock(
            block_type=OutputBlockType.TABLE,
            title="OLAP Cubes",
            content={
                "columns": columns,
                "rows": rows
            }
        )
    ]
    
    return NormalizedResult(
        analysis_type="olap_cubes",
        title="OLAP Cubes",
        variables={"summary": summary_vars, "grouping": grouping_vars},
        output_blocks=blocks,
        metadata={
            "n_total": len(df),
            "library": "pandas",
            "duration_ms": int((time.time() - start) * 1000),
            "timestamp": datetime.utcnow().isoformat(),
        }
    )

def run_report_summaries(df: pd.DataFrame, variables: list[str], orient: str = "rows") -> NormalizedResult:
    """Report Summaries in Rows or Columns."""
    start = time.time()
    
    valid_vars = [v for v in variables if v in df.columns]
    numeric_df = df[valid_vars].select_dtypes(include='number')
    
    if numeric_df.empty:
        raise ValueError("No numeric variables provided for report summaries.")
        
    summary = numeric_df.describe().round(3)
    
    if orient == "rows":
        # Statistics as rows
        summary_reset = summary.reset_index()
        summary_reset.rename(columns={"index": "Statistic"}, inplace=True)
        columns = list(summary_reset.columns)
        rows = summary_reset.to_dict(orient="records")
        title = "Report Summaries in Rows"
    else:
        # Statistics as columns
        summary_t = summary.T.reset_index()
        summary_t.rename(columns={"index": "Variable"}, inplace=True)
        columns = list(summary_t.columns)
        rows = summary_t.to_dict(orient="records")
        title = "Report Summaries in Columns"

    blocks = [
        OutputBlock(
            block_type=OutputBlockType.TABLE,
            title=title,
            content={
                "columns": columns,
                "rows": rows
            }
        )
    ]
    
    return NormalizedResult(
        analysis_type=f"report_{orient}",
        title=title,
        variables={"analyzed": variables},
        output_blocks=blocks,
        metadata={
            "n_total": len(df),
            "library": "pandas",
            "duration_ms": int((time.time() - start) * 1000),
            "timestamp": datetime.utcnow().isoformat(),
        }
    )

def run_custom_tables(df: pd.DataFrame, rows_vars: list[str] = None, cols_vars: list[str] = None) -> NormalizedResult:
    """Generate Custom Tables."""
    start = time.time()
    
    if not rows_vars and not cols_vars:
        raise ValueError("Custom Tables requires at least one row or column variable.")
        
    # Default to cross-tabulation logic if both provided
    if rows_vars and cols_vars:
        r_var = rows_vars[0]
        c_var = cols_vars[0]
        if r_var in df.columns and c_var in df.columns:
            ct = pd.crosstab(df[r_var], df[c_var], margins=True)
            
            # Format output
            columns = [r_var] + [str(c) for c in ct.columns]
            rows_data = []
            for idx, row in ct.iterrows():
                row_dict = {r_var: str(idx)}
                for col in ct.columns:
                    row_dict[str(col)] = int(row[col])
                rows_data.append(row_dict)
                
            blocks = [OutputBlock(
                block_type=OutputBlockType.TABLE,
                title=f"Custom Table: {r_var} x {c_var}",
                content={"columns": columns, "rows": rows_data}
            )]
        else:
            blocks = []
    else:
        # Fallback to simple summaries
        v_list = rows_vars or cols_vars
        v_list = [v for v in v_list if v in df.columns]
        summary = df[v_list].describe().round(3).reset_index()
        summary.rename(columns={"index": "Statistic"}, inplace=True)
        blocks = [OutputBlock(
            block_type=OutputBlockType.TABLE,
            title="Custom Table Summary",
            content={"columns": list(summary.columns), "rows": summary.to_dict(orient="records")}
        )]
        
    return NormalizedResult(
        analysis_type="custom_tables",
        title="Custom Tables",
        variables={"rows": rows_vars, "columns": cols_vars},
        output_blocks=blocks,
        metadata={
            "n_total": len(df),
            "library": "pandas",
            "duration_ms": int((time.time() - start) * 1000),
            "timestamp": datetime.utcnow().isoformat(),
        }
    )
