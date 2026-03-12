"""KALESS Engine — Analyze Request/Response Schemas."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class AnalyzeRequest(BaseModel):
    """Request to run a statistical analysis."""

    analysis_type: str = Field(..., description="Analysis type from registry")
    dataset_url: str = Field(..., description="Supabase storage path: bucket/path")
    file_type: str = Field(..., description="csv, xlsx, or tsv")
    params: dict[str, Any] = Field(default_factory=dict, description="Analysis parameters")
    encoding: str = Field("utf-8")
    delimiter: str | None = Field(None)


class AnalyzeResponse(BaseModel):
    """Response wrapper for an analysis result."""

    success: bool
    result: dict[str, Any] | None = None
    error: str | None = None
    error_code: str | None = None
