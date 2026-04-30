"""KALESS Engine — Export Routes (SAV, XLSX, PDF)."""

import io
import tempfile
import os
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, Response
from pydantic import BaseModel

from app.api.deps import verify_engine_key
from app.export.pdf_generator import generate_pdf
from app.export.docx_generator import generate_docx

router = APIRouter()


# ================================================================
# REQUEST SCHEMAS
# ================================================================

class ExportPdfRequest(BaseModel):
    result: Dict[str, Any]


class ExportFileRequest(BaseModel):
    """Request schema for SAV/XLSX export."""
    dataset_url: str
    file_type: str = "csv"
    columns: Optional[List[str]] = None
    dataset_name: Optional[str] = "kaless_export"


# ================================================================
# PDF/DOCX EXPORT (Output Viewer)
# ================================================================

@router.post("/export/pdf")
async def export_pdf_endpoint(
    request: ExportPdfRequest,
    _key: str = Depends(verify_engine_key),
):
    """Generate an A4 PDF from a NormalizedResult payload."""
    try:
        pdf_bytes = generate_pdf(request.result)
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={
                "Content-Disposition": "attachment; filename=kaless_report.pdf"
            }
        )
    except Exception as e:
        return Response(content=f"PDF Generation failed: {str(e)}", status_code=500)

@router.post("/export/docx")
async def export_docx_endpoint(
    request: ExportPdfRequest, # Reusing the same request schema as PDF since payload is identical
    _key: str = Depends(verify_engine_key),
):
    """Generate an APA-formatted Docx from a NormalizedResult payload."""
    try:
        docx_bytes = generate_docx(request.result)
        return Response(
            content=docx_bytes,
            media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            headers={
                "Content-Disposition": "attachment; filename=kaless_report.docx"
            }
        )
    except Exception as e:
        return Response(content=f"Docx Generation failed: {str(e)}", status_code=500)


# ================================================================
# HELPERS
# ================================================================

def _load_dataframe(dataset_url: str, file_type: str):
    """Load a dataset into a Pandas DataFrame from various formats."""
    import pandas as pd

    if file_type == "parquet":
        return pd.read_parquet(dataset_url)
    elif file_type == "csv":
        return pd.read_csv(dataset_url, encoding="utf-8")
    elif file_type in ("xlsx", "xls"):
        return pd.read_excel(dataset_url)
    elif file_type == "tsv":
        return pd.read_csv(dataset_url, sep="\t", encoding="utf-8")
    else:
        return pd.read_csv(dataset_url, encoding="utf-8")


# ================================================================
# SAV EXPORT (.sav — SPSS format via pyreadstat)
# ================================================================

@router.post("/export/sav")
async def export_sav_endpoint(
    request: ExportFileRequest,
    _key: str = Depends(verify_engine_key),
):
    """Export dataset as SPSS .sav file with full UTF-8 support."""
    try:
        import pyreadstat
        import pandas as pd

        df = _load_dataframe(request.dataset_url, request.file_type)

        # Filter columns if specified
        if request.columns:
            valid_cols = [c for c in request.columns if c in df.columns]
            if valid_cols:
                df = df[valid_cols]

        # Sanitize column names for SPSS compatibility (max 64 chars, no special chars)
        clean_names = {}
        for col in df.columns:
            safe = str(col).replace(" ", "_").replace("-", "_")
            safe = "".join(c for c in safe if c.isalnum() or c == "_")
            if not safe or not safe[0].isalpha():
                safe = "V_" + safe
            safe = safe[:64]
            clean_names[col] = safe
        df = df.rename(columns=clean_names)

        # Build column labels (original names preserved as labels)
        column_labels = {v: k for k, v in clean_names.items()}

        # Ensure string columns are properly encoded
        for col in df.select_dtypes(include=["object"]).columns:
            df[col] = df[col].astype(str).replace("nan", "")

        # Write to temporary .sav file
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".sav")
        tmp_path = tmp.name
        tmp.close()

        try:
            pyreadstat.write_sav(
                df,
                tmp_path,
                column_labels=column_labels,
                file_label=f"KALESS Export: {request.dataset_name}",
            )

            with open(tmp_path, "rb") as f:
                sav_bytes = f.read()

            filename = f"{request.dataset_name or 'export'}.sav"
            return Response(
                content=sav_bytes,
                media_type="application/octet-stream",
                headers={
                    "Content-Disposition": f'attachment; filename="{filename}"'
                }
            )
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    except ImportError:
        return Response(
            content='{"error": "pyreadstat is not installed. Run: pip install pyreadstat"}',
            status_code=500,
            media_type="application/json",
        )
    except Exception as e:
        return Response(
            content=f'{{"error": "SAV export failed: {str(e)}"}}',
            status_code=500,
            media_type="application/json",
        )


# ================================================================
# XLSX EXPORT (Multi-sheet Excel via openpyxl)
# ================================================================

@router.post("/export/xlsx")
async def export_xlsx_endpoint(
    request: ExportFileRequest,
    _key: str = Depends(verify_engine_key),
):
    """Export dataset as multi-sheet Excel file."""
    try:
        import pandas as pd

        df = _load_dataframe(request.dataset_url, request.file_type)

        # Filter columns if specified
        if request.columns:
            valid_cols = [c for c in request.columns if c in df.columns]
            if valid_cols:
                df = df[valid_cols]

        buffer = io.BytesIO()

        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            # Sheet 1: Data
            df.to_excel(writer, sheet_name="Data", index=False)

            # Sheet 2: Variable Metadata
            meta_rows = []
            for i, col in enumerate(df.columns):
                dtype = str(df[col].dtype)
                non_null = int(df[col].notna().sum())
                null_count = int(df[col].isna().sum())
                unique_count = int(df[col].nunique())

                measure = "Scale" if pd.api.types.is_numeric_dtype(df[col]) else "Nominal"

                meta_rows.append({
                    "Position": i + 1,
                    "Variable": col,
                    "Type": dtype,
                    "Measure": measure,
                    "Valid_N": non_null,
                    "Missing_N": null_count,
                    "Unique": unique_count,
                })
            meta_df = pd.DataFrame(meta_rows)
            meta_df.to_excel(writer, sheet_name="Variable Info", index=False)

            # Sheet 3: Descriptive Statistics (numeric columns only)
            numeric_df = df.select_dtypes(include=["number"])
            if not numeric_df.empty:
                desc = numeric_df.describe().T
                desc.index.name = "Variable"
                desc.to_excel(writer, sheet_name="Descriptives")

        xlsx_bytes = buffer.getvalue()
        filename = f"{request.dataset_name or 'export'}.xlsx"

        return Response(
            content=xlsx_bytes,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"'
            }
        )

    except Exception as e:
        return Response(
            content=f'{{"error": "XLSX export failed: {str(e)}"}}',
            status_code=500,
            media_type="application/json",
        )
