"""KALESS Engine — Parse Request/Response Schemas."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ParseRequest(BaseModel):
    """Request to parse a dataset file."""

    file_url: str = Field(..., description="Supabase storage path: bucket/path")
    file_type: str = Field(..., description="csv, xlsx, or tsv")
    encoding: str = Field("utf-8", description="Character encoding")
    delimiter: str | None = Field(None, description="Delimiter override for CSV")
    max_preview_rows: int = Field(100, ge=10, le=500)


class ColumnSchema(BaseModel):
    """Inferred column metadata."""

    index: int
    name: str
    inferred_type: str
    measure_level: str
    missing_count: int
    unique_count: int
    sample_values: list[Any] = []


class ParseResponse(BaseModel):
    """Response from parsing a dataset."""

    row_count: int
    column_count: int
    columns: list[ColumnSchema]
    preview_rows: list[dict[str, Any]]
    warnings: list[str] = []
    encoding: str
    delimiter: str | None = None
    parquet_path: str | None = None
