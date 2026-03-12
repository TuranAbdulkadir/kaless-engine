"""KALESS Engine — Analyze Route (statistical analysis execution)."""

from __future__ import annotations

import io

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException

from app.api.deps import verify_engine_key
from app.analysis.registry import dispatch_analysis, get_available_analyses
from app.core.parser import _read_file
from app.schemas.analyze import AnalyzeRequest, AnalyzeResponse
from app.utils.errors import KalessEngineError
from app.utils.storage import download_file

router = APIRouter()


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze_endpoint(
    request: AnalyzeRequest,
    _key: str = Depends(verify_engine_key),
):
    """Execute a statistical analysis on a dataset.

    Downloads the dataset, loads it into a DataFrame, dispatches
    to the requested analysis module, and returns NormalizedResult.
    """
    # Download file
    try:
        parts = request.dataset_url.split("/", 1)
        if len(parts) != 2:
            raise HTTPException(status_code=400, detail="Invalid dataset_url format.")
        bucket, path = parts
        file_content = download_file(bucket, path)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=502,
            detail=f"Failed to download dataset: {str(e)}",
        )

    # Load into DataFrame
    try:
        warnings: list[str] = []
        df = _read_file(
            file_content,
            request.file_type,
            request.encoding,
            request.delimiter,
            warnings,
        )
    except Exception as e:
        return AnalyzeResponse(
            success=False,
            error=f"Failed to load dataset: {str(e)}",
            error_code="LOAD_ERROR",
        )

    # Dispatch analysis
    try:
        result = dispatch_analysis(
            analysis_type=request.analysis_type,
            df=df,
            params=request.params,
        )
        return AnalyzeResponse(
            success=True,
            result=result.model_dump(),
        )
    except KalessEngineError as e:
        return AnalyzeResponse(
            success=False,
            error=e.message,
            error_code=e.code,
        )
    except Exception as e:
        return AnalyzeResponse(
            success=False,
            error=f"Unexpected analysis error: {str(e)}",
            error_code="INTERNAL_ERROR",
        )


@router.get("/analyses")
async def list_analyses(
    _key: str = Depends(verify_engine_key),
):
    """List available analysis types with their parameters."""
    return {"analyses": get_available_analyses()}
