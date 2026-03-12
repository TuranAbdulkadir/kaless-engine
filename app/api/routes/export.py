"""KALESS Engine — Export Routes."""

from fastapi import APIRouter, Depends, Response
from pydantic import BaseModel
from typing import Any, Dict

from app.api.deps import verify_engine_key
from app.export.pdf_generator import generate_pdf

router = APIRouter()

class ExportPdfRequest(BaseModel):
    result: Dict[str, Any]

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
                "Content-Disposition": f"attachment; filename=kaless_report.pdf"
            }
        )
    except Exception as e:
        return Response(content=f"PDF Generation failed: {str(e)}", status_code=500)
