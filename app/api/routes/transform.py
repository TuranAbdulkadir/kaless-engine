"""KALESS Engine — Transform Route."""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
import io
import time
from typing import Any

from app.api.deps import verify_engine_key
from app.core.parser import parse_dataset
from app.utils.errors import ParseError
from app.utils.storage import download_file, upload_file
from app.transforms import operations

router = APIRouter()

class TransformRequest(BaseModel):
    file_url: str = Field(..., description="Supabase storage path of the current dataset version")
    file_type: str = Field("parquet", description="Current file format, ideally parquet")
    transform_type: str = Field(..., description="E.g., compute, z_score, recode, reverse_code, filter, sort")
    params: dict[str, Any] = Field(..., description="Parameters for the specific transformation")

class TransformResponse(BaseModel):
    success: bool
    new_file_url: str | None = None
    row_count: int | None = None
    column_count: int | None = None
    columns: list[dict[str, Any]] | None = None
    preview_rows: list[dict[str, Any]] | None = None
    error: str | None = None

class ComputeJSONRequest(BaseModel):
    raw_data: list[dict[str, Any]] = Field(..., description="The dataset as a list of dictionaries")
    target_col: str = Field(..., description="Name of the new variable to compute")
    expression: str = Field(..., description="The mathematical expression (e.g. 'SCORE * 2')")

@router.post("/transform/compute")
async def compute_json_endpoint(
    request: ComputeJSONRequest,
    _key: str = Depends(verify_engine_key),
):
    try:
        import pandas as pd
        import numpy as np
        
        if not request.raw_data:
            return {"success": False, "error": "No data provided."}
            
        df = pd.DataFrame(request.raw_data)
        
        # Use pandas eval for vectorized computation. 
        # Support basic python engine fallback if needed.
        try:
            df[request.target_col] = pd.eval(request.expression, target=df)
        except Exception:
            # Fallback for some mathematical string expressions
            df[request.target_col] = df.eval(request.expression)
            
        # Clean up NaNs so JSON serialization doesn't fail
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.fillna(value="", inplace=True)
        
        return {"success": True, "result": df.to_dict(orient="records")}
    except Exception as e:
        return {"success": False, "error": f"Computation failed: {str(e)}"}

class AddVariableRequest(BaseModel):
    file_url: str
    file_type: str = "parquet"
    target_col: str
    default_value: Any = None

@router.post("/dataset/add_variable")
async def add_variable_endpoint(
    request: AddVariableRequest,
    _key: str = Depends(verify_engine_key),
):
    try:
        bucket, path = request.file_url.split("/", 1)
        file_content = download_file(bucket, path)
        warnings = []
        from app.core.parser import _read_file
        df = _read_file(content=file_content, file_type=request.file_type, encoding="utf-8", delimiter=None, warnings=warnings)
        
        df = operations.add_variable(df, request.target_col, request.default_value)
        
        parquet_buf = io.BytesIO()
        df.to_parquet(parquet_buf, index=False)
        parquet_bytes = parquet_buf.getvalue()
        
        base = path.rsplit(".", 1)[0]
        if "_v" in base:
           base = base.rsplit("_v", 1)[0]
        new_path = f"{base}_v{int(time.time())}.parquet"
        
        upload_file(bucket, new_path, parquet_bytes, "application/octet-stream")
        
        from app.core.parser import parse_dataset
        result = parse_dataset(file_content=parquet_bytes, file_type="parquet")
        
        return {
            "success": True,
            "new_file_url": f"{bucket}/{new_path}",
            "row_count": result["row_count"],
            "column_count": result["column_count"],
            "columns": result["columns"],
            "preview_rows": result["preview_rows"]
        }
    except Exception as e:
        return {"success": False, "error": f"Failed to add variable: {str(e)}"}

@router.post("/transform", response_model=TransformResponse)
async def transform_endpoint(
    request: TransformRequest,
    _key: str = Depends(verify_engine_key),
):
    try:
        parts = request.file_url.split("/", 1)
        if len(parts) != 2:
            raise HTTPException(status_code=400, detail="Invalid file_url format.")
        bucket, path = parts
        file_content = download_file(bucket, path)
    except Exception as e:
        return TransformResponse(success=False, error=f"Failed to download payload: {str(e)}")

    try:
        # We parse the dataset initially just to get the df. We can bypass parsing metadata until the end.
        warnings = []
        from app.core.parser import _read_file
        df = _read_file(content=file_content, file_type=request.file_type, encoding="utf-8", delimiter=None, warnings=warnings)
    except Exception as e:
        return TransformResponse(success=False, error=f"Failed to load dataset: {str(e)}")

    try:
        # Dispatch Operation
        op = request.transform_type
        p = request.params
        if op == "compute":
            df = operations.compute_variable(df, p["target_col"], p["expression"])
        elif op == "z_score":
            df = operations.z_score(df, p["columns"])
        elif op == "recode":
            df = operations.recode(df, p["column"], p["target_col"], p["rules"], p.get("default_value"))
        elif op == "reverse_code":
            df = operations.reverse_code(df, p["columns"], p["min_val"], p["max_val"])
        elif op == "filter":
            df = operations.filter_cases(df, p["condition"])
        elif op == "sort":
            df = operations.sort_cases(df, p["column"], p.get("ascending", True))
        elif op == "transpose":
            df = operations.transpose_dataset(df)
        elif op == "rank":
            df = operations.rank_cases(df, p["column"], p["target_col"], p.get("ascending", True))
        elif op == "count":
            df = operations.count_values(df, p["target_col"], p["columns"], p["value_to_count"])
        elif op == "merge":
            # For merge, we need to download the second dataset
            parts2 = p["file_url_2"].split("/", 1)
            file_content_2 = download_file(parts2[0], parts2[1])
            df2 = _read_file(content=file_content_2, file_type=p.get("file_type_2", "parquet"), encoding="utf-8", delimiter=None, warnings=[])
            df = operations.merge_datasets(df, df2, p["merge_type"], p.get("key_col"))
        elif op == "automatic_recode":
            df = operations.automatic_recode(df, p["column"], p["target_col"])
        elif op == "visual_binning":
            df = operations.visual_binning(df, p["column"], p["target_col"], p["cutpoints"], p.get("labels"))
        elif op == "add_variable":
            df = operations.add_variable(df, p["target_col"], p.get("default_value"))
        else:
            return TransformResponse(success=False, error=f"Unsupported transform logic: {op}")
            
    except Exception as e:
        return TransformResponse(success=False, error=f"Transform operation failed: {str(e)}")

    try:
        # Save new canonical parquet
        parquet_buf = io.BytesIO()
        df.to_parquet(parquet_buf, index=False)
        parquet_bytes = parquet_buf.getvalue()
        
        # New filename strategy: insert timestamp right before .parquet
        base = path.rsplit(".", 1)[0]
        # remove old timestamp if it had one like _v1234
        if "_v" in base:
           base = base.rsplit("_v", 1)[0]
        new_path = f"{base}_v{int(time.time())}.parquet"
        
        upload_file(bucket, new_path, parquet_bytes, "application/octet-stream")
        
        # We need to re-run the parser to get the fresh metadata for the frontend
        result = parse_dataset(
            file_content=parquet_bytes,
            file_type="parquet"
        )
        
        return TransformResponse(
            success=True,
            new_file_url=f"{bucket}/{new_path}",
            row_count=result["row_count"],
            column_count=result["column_count"],
            columns=result["columns"],
            preview_rows=result["preview_rows"]
        )
        
    except Exception as e:
        return TransformResponse(success=False, error=f"Failed to save engineered dataset: {str(e)}")
