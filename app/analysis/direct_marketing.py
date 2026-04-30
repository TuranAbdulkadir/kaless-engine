"""Direct Marketing (RFM Analysis)."""

import pandas as pd
import numpy as np
from app.schemas.results import NormalizedResult, OutputBlock

def run_direct_marketing(df: pd.DataFrame, customer_id: str, date_var: str, monetary_var: str) -> NormalizedResult:
    """Run RFM (Recency, Frequency, Monetary) Analysis."""
    required_cols = [customer_id, date_var, monetary_var]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in dataset.")

    df_clean = df.dropna(subset=required_cols).copy()

    if len(df_clean) == 0:
        raise ValueError("No valid cases found after dropping missing values.")

    # Convert date column
    try:
        df_clean[date_var] = pd.to_datetime(df_clean[date_var])
    except Exception:
        raise ValueError(f"Column '{date_var}' could not be parsed as a date.")

    ref_date = df_clean[date_var].max() + pd.Timedelta(days=1)

    rfm = df_clean.groupby(customer_id).agg({
        date_var: lambda x: (ref_date - x.max()).days,      # Recency
        customer_id: "count",                                 # Frequency (using customer_id as proxy)
        monetary_var: "sum"                                   # Monetary
    })

    # Rename columns to avoid collision
    rfm.columns = ["Recency", "Frequency", "Monetary"]
    rfm = rfm.reset_index()

    # Scoring: quintile-based (1-5)
    for col in ["Recency", "Frequency", "Monetary"]:
        try:
            if col == "Recency":
                rfm[f"{col}_Score"] = pd.qcut(rfm[col], q=5, labels=[5, 4, 3, 2, 1], duplicates="drop").astype(int)
            else:
                rfm[f"{col}_Score"] = pd.qcut(rfm[col], q=5, labels=[1, 2, 3, 4, 5], duplicates="drop").astype(int)
        except Exception:
            rfm[f"{col}_Score"] = 3  # Fallback if not enough unique values

    rfm["RFM_Score"] = rfm["Recency_Score"] + rfm["Frequency_Score"] + rfm["Monetary_Score"]

    # Summary statistics
    summary_rows = [
        {"Metric": "Total Customers", "Value": str(len(rfm))},
        {"Metric": "Avg Recency (days)", "Value": f"{rfm['Recency'].mean():.1f}"},
        {"Metric": "Avg Frequency", "Value": f"{rfm['Frequency'].mean():.1f}"},
        {"Metric": "Avg Monetary", "Value": f"{rfm['Monetary'].mean():.2f}"},
        {"Metric": "Avg RFM Score", "Value": f"{rfm['RFM_Score'].mean():.1f}"},
    ]

    # Top 10 customers
    top10 = rfm.nlargest(10, "RFM_Score")
    top10_rows = []
    for _, row in top10.iterrows():
        top10_rows.append({
            "Customer": str(row[customer_id]),
            "Recency": str(row["Recency"]),
            "Frequency": str(row["Frequency"]),
            "Monetary": f"{row['Monetary']:.2f}",
            "RFM Score": str(row["RFM_Score"])
        })

    output_blocks = [
        OutputBlock(
            block_type=OutputBlockType.TABLE,
            title="RFM Summary Statistics",
            content={
                "columns": ["Metric", "Value"],
                "rows": summary_rows,
                "footnotes": ["RFM = Recency + Frequency + Monetary (quintile scores 1-5)"]
            }
        ),
        OutputBlock(
            block_type=OutputBlockType.TABLE,
            title="Top 10 Customers by RFM Score",
            content={
                "columns": ["Customer", "Recency", "Frequency", "Monetary", "RFM Score"],
                "rows": top10_rows,
                "footnotes": []
            }
        )
    ]

    return NormalizedResult(
        title="Direct Marketing — RFM Analysis",
        variables={"analyzed": required_cols},
        output_blocks=output_blocks,
        interpretation={"academic_sentence": "An RFM (Recency, Frequency, Monetary) analysis was conducted to segment customers."}
    )
