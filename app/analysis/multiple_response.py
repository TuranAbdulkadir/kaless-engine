"""Multiple Response Analysis."""

import pandas as pd
import numpy as np
from app.schemas.results import NormalizedResult, OutputBlock

def run_multiple_response(df: pd.DataFrame, variables: list[str], count_value: str = "1") -> NormalizedResult:
    """Run Multiple Response Frequencies."""
    if not variables:
        raise ValueError("Multiple response analysis requires at least one variable.")
        
    df_sub = df[variables]
    
    # Check what type of data we have to cast count_value properly
    try:
        count_val_num = float(count_value)
    except ValueError:
        count_val_num = None
        
    freq_data = []
    total_responses = 0
    total_cases = len(df_sub)
    
    # Find how many valid cases have AT LEAST ONE response
    valid_cases_mask = pd.Series([False] * total_cases, index=df_sub.index)
    
    for col in variables:
        if pd.api.types.is_numeric_dtype(df_sub[col]) and count_val_num is not None:
            mask = (df_sub[col] == count_val_num)
        else:
            mask = (df_sub[col].astype(str) == str(count_value))
            
        count = mask.sum()
        total_responses += count
        valid_cases_mask = valid_cases_mask | mask
        
        freq_data.append({
            "Name": col,
            "Count": count
        })
        
    valid_cases = valid_cases_mask.sum()
    
    rows = []
    for d in freq_data:
        pct_responses = (d["Count"] / total_responses * 100) if total_responses > 0 else 0
        pct_cases = (d["Count"] / valid_cases * 100) if valid_cases > 0 else 0
        rows.append({
            "Name": d["Name"],
            "Responses N": str(d["Count"]),
            "Responses Percent": f"{pct_responses:.1f}%",
            "Percent of Cases": f"{pct_cases:.1f}%"
        })
        
    # Add Total row
    rows.append({
        "Name": "Total",
        "Responses N": str(total_responses),
        "Responses Percent": "100.0%",
        "Percent of Cases": f"{(total_responses / valid_cases * 100) if valid_cases > 0 else 0:.1f}%"
    })
        
    output_blocks = [
        OutputBlock(
            block_type=OutputBlockType.TABLE,
            title="Multiple Response Frequencies",
            content={
                "columns": ["Name", "Responses N", "Responses Percent", "Percent of Cases"],
                "rows": rows,
                "footnotes": [f"Value counted: {count_value}", f"Valid Cases: {valid_cases}"]
            }
        )
    ]

    return NormalizedResult(
        title="Multiple Response Analysis",
        variables={"analyzed": variables},
        output_blocks=output_blocks,
        interpretation={"academic_sentence": "A multiple response analysis was conducted on the dataset."}
    )
