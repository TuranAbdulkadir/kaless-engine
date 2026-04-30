"""Forecasting (Time Series)."""

import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from app.schemas.results import NormalizedResult, OutputBlock, ChartData, OutputBlockType

def run_forecasting(df: pd.DataFrame, dependent: str, date_var: str = None, steps: int = 10) -> NormalizedResult:
    """Run ARIMA Forecasting."""
    df_clean = df.dropna(subset=[dependent]).copy()
    
    if len(df_clean) < 10:
        raise ValueError("Forecasting requires at least 10 valid cases.")
        
    y = df_clean[dependent].values
    
    try:
        model = ARIMA(y, order=(1, 1, 1))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=steps)
    except Exception as e:
        raise ValueError(f"ARIMA model failed to converge: {str(e)}")
        
    output_blocks = [
        OutputBlock(
            block_type=OutputBlockType.TABLE,
            title="Model Summary",
            content={
                "columns": ["Model", "AIC", "BIC"],
                "rows": [
                    {"Model": "ARIMA(1, 1, 1)", "AIC": f"{model_fit.aic:.3f}", "BIC": f"{model_fit.bic:.3f}"}
                ],
                "footnotes": ["Time Series Modeler"]
            }
        ),
        OutputBlock(
            block_type=OutputBlockType.TABLE,
            title="Forecasts",
            content={
                "columns": ["Step", "Forecast"],
                "rows": [{"Step": str(i+1), "Forecast": f"{val:.3f}"} for i, val in enumerate(forecast)],
                "footnotes": []
            }
        )
    ]
    
    # Mock chart data combining actuals and forecast
    chart_data = []
    for i, val in enumerate(y[-20:]): # Show last 20 actuals
        chart_data.append({"Time": f"T-{20-i}", "Actual": float(val), "Forecast": None})
    for i, val in enumerate(forecast):
        chart_data.append({"Time": f"F+{i+1}", "Actual": None, "Forecast": float(val)})
        
    charts = [
        ChartData(
            chart_type="line",
            data=chart_data,
            config={"title": f"Forecast for {dependent}", "x_axis": "Time", "y_axis": "Value"}
        )
    ]

    return NormalizedResult(
        title="Time Series Forecasting",
        variables={"analyzed": [dependent]},
        output_blocks=output_blocks,
        charts=charts,
        interpretation={"academic_sentence": f"An ARIMA(1,1,1) model was fitted to forecast {steps} future values of {dependent}."}
    )
