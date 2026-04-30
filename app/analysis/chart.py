import pandas as pd
from typing import Any
import time
import numpy as np

from app.schemas.results import (
    NormalizedResult,
    OutputBlock,
    OutputBlockType,
    Interpretation
)
from app.utils.errors import ValidationError

def run_chart_builder(df: pd.DataFrame, params: dict[str, Any]) -> NormalizedResult:
    """Generates JSON structural data for React Recharts to draw SPSS-style charts."""
    start_time = time.time()
    
    chart_type = params.get("chart_type", "bar")
    x_axis = params.get("x_axis")
    y_axis = params.get("y_axis")
    
    if not x_axis or x_axis not in df.columns:
        raise ValidationError(f"Valid x_axis is required. Given: {x_axis}")
        
    data_list = []
    config = {"x_label": x_axis, "y_label": y_axis or "Count"}
    
    # Drop completely empty rows for the relevant variables
    cols_to_check = [x_axis]
    if y_axis and y_axis in df.columns:
        cols_to_check.append(y_axis)
    df_clean = df.dropna(subset=cols_to_check).copy()
    
    # Process data based on chart type
    if chart_type in ["bar", "pie"]:
        # Frequencies for categorical
        counts = df_clean[x_axis].value_counts().reset_index()
        counts.columns = [x_axis, "count"]
        # Convert to records
        for _, row in counts.iterrows():
            data_list.append({"name": str(row[x_axis]), "value": int(row["count"])})
            
    elif chart_type == "histogram":
        # Bin numeric data
        if not pd.api.types.is_numeric_dtype(df_clean[x_axis]):
            # Attempt conversion
            df_clean[x_axis] = pd.to_numeric(df_clean[x_axis], errors="coerce")
            df_clean = df_clean.dropna(subset=[x_axis])
            if df_clean.empty:
                raise ValidationError(f"Histogram requires numeric x_axis. Could not convert '{x_axis}' to numeric.")
        
        # Prevent zero-variance error in pd.cut
        if df_clean[x_axis].nunique() <= 1:
            data_list.append({"name": str(df_clean[x_axis].iloc[0]), "value": len(df_clean)})
        else:
            hist, bin_edges = pd.cut(df_clean[x_axis], bins=15, retbins=True, include_lowest=True)
            counts = hist.value_counts(sort=False)
            for interval, count in counts.items():
                # Center of bin for cleaner axis labels
                mid = (interval.left + interval.right) / 2
                name_str = f"{mid:.1f}" if abs(mid) < 1000 else f"{int(mid)}"
                data_list.append({"name": name_str, "value": int(count)})
            
    elif chart_type == "scatter":
        if not y_axis or y_axis not in df.columns:
            raise ValidationError("Scatter plot requires both x_axis and y_axis.")
            
        # Ensure numeric
        df_clean[x_axis] = pd.to_numeric(df_clean[x_axis], errors="coerce")
        df_clean[y_axis] = pd.to_numeric(df_clean[y_axis], errors="coerce")
        df_clean = df_clean.dropna(subset=[x_axis, y_axis])
        
        for _, row in df_clean.iterrows():
            data_list.append({"x": float(row[x_axis]), "y": float(row[y_axis])})
            
    elif chart_type == "line":
        # Average y_axis by x_axis, or just plot series
        if y_axis:
            df_clean[y_axis] = pd.to_numeric(df_clean[y_axis], errors="coerce")
            agg_df = df_clean.groupby(x_axis)[y_axis].mean().reset_index()
            for _, row in agg_df.iterrows():
                data_list.append({"name": str(row[x_axis]), "value": float(row[y_axis])})
        else:
             # Just frequency like bar
            counts = df_clean[x_axis].value_counts().sort_index().reset_index()
            counts.columns = [x_axis, "count"]
            for _, row in counts.iterrows():
                data_list.append({"name": str(row[x_axis]), "value": int(row["count"])})
                
    else:
        raise ValidationError(f"Unsupported chart type: {chart_type}")
        
    title_map = {
        "bar": "Bar Chart",
        "pie": "Pie Chart",
        "histogram": "Histogram",
        "scatter": "Scatter Plot",
        "line": "Line Chart"
    }
    title = title_map.get(chart_type, "Graph")
    
    # To match SPSS, we don't just return charts—it goes into output_blocks
    blocks = [
        OutputBlock(
            block_type=OutputBlockType.CHART,
            title=title,
            display_order=1,
            content={
                "chart_type": chart_type,
                "data": data_list,
                "config": config
            }
        )
    ]
    
    return NormalizedResult(
        analysis_type="chart_builder",
        title=f"GGraph - {title}",
        variables={"x_axis": x_axis, "y_axis": y_axis or ""},
        output_blocks=blocks,
        interpretation=Interpretation(
            summary=f"Generated a {chart_type} for '{x_axis}'.",
            academic_sentence=f"A {chart_type} was constructed to visualize the distribution of {x_axis}{' and ' + y_axis if y_axis else ''}."
        ),
        metadata={
            "n_total": len(df),
            "missing_excluded": len(df) - len(df_clean),
            "library": "pandas",
            "duration_ms": int((time.time() - start_time) * 1000),
            "timestamp": pd.Timestamp.utcnow().isoformat()
        }
    )
