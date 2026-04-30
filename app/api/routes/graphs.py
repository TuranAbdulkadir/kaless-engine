"""KALESS Engine — Graph Route (chart data generation)."""

import pandas as pd
import numpy as np
import scipy.stats as stats
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import Any

from app.api.deps import verify_engine_key
from app.schemas.results import NormalizedResult, OutputBlock, OutputBlockType, ChartData


router = APIRouter()

class GraphRequest(BaseModel):
    chart_type: str
    dataset_url: str | None = None
    raw_data: list[dict[str, Any]] | None = None
    file_type: str = "csv"
    variables: list[str] = []
    x_axis: str | None = None
    y_axis: str | None = None
    grouping_var: str | None = None
    display_normal_curve: bool = False
    chart_title: str | None = None


@router.post("/graphs")
async def generate_chart_data(
    req: GraphRequest,
    _key: str = Depends(verify_engine_key),
):
    """Generate chart-ready data for Recharts."""
    try:
        # Load data
        if req.raw_data is not None:
            if isinstance(req.raw_data, list):
                df = pd.DataFrame(req.raw_data)
            else:
                df = pd.DataFrame.from_dict(req.raw_data, orient='columns')
        else:
            raise HTTPException(status_code=400, detail="raw_data must be provided.")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    if df.empty:
        raise HTTPException(status_code=400, detail="Dataset is empty.")

    chart_type = req.chart_type.lower()
    x_axis = req.x_axis
    y_axis = req.y_axis

    chart_data = []
    config = {}

    if chart_type in ["bar", "pie"]:
        if not x_axis:
            raise HTTPException(status_code=400, detail="X-Axis is required for bar/pie charts.")
        # Simple count per category
        counts = df[x_axis].value_counts().reset_index()
        counts.columns = ['name', 'value']
        
        # Sort categorical strings
        counts = counts.sort_values(by='name')
        
        # Convert to list of dicts
        chart_data = counts.to_dict('records')
        
        # Ensure values are JSON serializable
        for item in chart_data:
            item['name'] = str(item['name'])
            item['value'] = float(item['value'])

    elif chart_type == "scatter":
        if not x_axis or not y_axis:
            raise HTTPException(status_code=400, detail="X-Axis and Y-Axis are required for scatter plot.")
        
        # Drop missing
        sub_df = df[[x_axis, y_axis]].dropna()
        
        if req.grouping_var and req.grouping_var in df.columns:
            sub_df[req.grouping_var] = df[req.grouping_var]
            
        chart_data = sub_df.to_dict('records')

    elif chart_type == "histogram":
        if not x_axis:
            raise HTTPException(status_code=400, detail="Variable is required for histogram.")
            
        data = df[x_axis].dropna()
        if len(data) == 0:
            raise HTTPException(status_code=400, detail="No valid numeric data for histogram.")

        # Calculate bins
        counts, bin_edges = np.histogram(data, bins='auto')
        
        # Create Recharts data
        for i in range(len(counts)):
            bin_center = (bin_edges[i] + bin_edges[i+1]) / 2
            item = {
                'bin': f"{bin_edges[i]:.2f} - {bin_edges[i+1]:.2f}",
                'count': int(counts[i]),
                'x': float(bin_center)
            }
            chart_data.append(item)
            
        # Add normal curve if requested
        if req.display_normal_curve:
            mean = data.mean()
            std = data.std()
            n = len(data)
            
            # The height of the normal curve over a histogram with counts
            # is PDF * N * bin_width
            bin_width = bin_edges[1] - bin_edges[0]
            for item in chart_data:
                x_val = item['x']
                pdf_val = stats.norm.pdf(x_val, mean, std)
                expected_count = pdf_val * n * bin_width
                item['normalCurve'] = float(expected_count)
                
            config['showNormalCurve'] = True

    elif chart_type == "boxplot":
        if not y_axis:
            raise HTTPException(status_code=400, detail="Scale variable (Y-Axis) is required for Boxplot.")
            
        # Optional X axis for grouped boxplots
        groups = [None]
        if x_axis:
            groups = df[x_axis].dropna().unique().tolist()
            groups = sorted([str(g) for g in groups])
            
        for g in groups:
            if g is None:
                series = df[y_axis].dropna()
                name = "Overall"
            else:
                series = df[df[x_axis].astype(str) == g][y_axis].dropna()
                name = g
                
            if len(series) == 0:
                continue
                
            q1 = series.quantile(0.25)
            median = series.quantile(0.50)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            
            lower_fence = q1 - 1.5 * iqr
            upper_fence = q3 + 1.5 * iqr
            
            # Whiskers are the min/max values within the fences
            valid_vals = series[(series >= lower_fence) & (series <= upper_fence)]
            min_whisker = valid_vals.min() if len(valid_vals) > 0 else q1
            max_whisker = valid_vals.max() if len(valid_vals) > 0 else q3
            
            # Outliers are points outside fences
            outliers = series[(series < lower_fence) | (series > upper_fence)].tolist()
            
            item = {
                'name': name,
                'min': float(min_whisker),
                'q1': float(q1),
                'median': float(median),
                'q3': float(q3),
                'max': float(max_whisker),
                'outliers': [float(o) for o in outliers]
            }
            chart_data.append(item)

    else:
        raise HTTPException(status_code=400, detail=f"Unsupported chart type: {chart_type}")

    # Build the NormalizedResult
    chart_block = ChartData(
        chart_type=chart_type,
        data=chart_data,
        config=config
    )

    result = NormalizedResult(
        analysis_type=f"graph_{chart_type}",
        title=req.chart_title or f"{chart_type.capitalize()} Chart",
        variables={
            "X-Axis": req.x_axis or "None",
            "Y-Axis": req.y_axis or "None",
            "Grouping": req.grouping_var or "None"
        },
        charts=[chart_block],
        output_blocks=[
            OutputBlock(
                block_type=OutputBlockType.CHART,
                title=req.chart_title or f"{chart_type.capitalize()} Chart",
                content=chart_block.dict(),
                display_order=1
            )
        ]
    )

    return result
