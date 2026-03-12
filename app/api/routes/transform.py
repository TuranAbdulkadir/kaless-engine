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
            df = operations.recode(df, p["column"], p["target_col"], p["mapping"], p.get("default_value"))
        elif op == "reverse_code":
            df = operations.reverse_code(df, p["columns"], p["min_val"], p["max_val"])
        elif op == "filter":
            df = operations.filter_cases(df, p["condition"])
        elif op == "sort":
            df = operations.sort_cases(df, p["column"], p.get("ascending", True))
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
