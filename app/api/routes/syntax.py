from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import Any, Optional

from app.api.deps import verify_engine_key

from app.core.syntax_processor import parse_syntax_command
from app.api.routes.analyze import AnalyzeRequest

router = APIRouter(prefix="/syntax", tags=["syntax"])

class SyntaxRequest(BaseModel):
    syntax: str
    dataset_url: Optional[str] = None
    raw_data: Optional[list[dict]] = None
    file_type: str = "csv"

@router.post("")
async def process_syntax(
    request: SyntaxRequest,
    _key: str = Depends(verify_engine_key),
):
    """Parses a syntax string and dispatches the underlying analysis command."""
    try:
        # 1. Parse the string
        plan = parse_syntax_command(request.syntax)
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "error_code": "SYNTAX_ERROR"
        }
        
    # 2. Delegate to /analyze logic via internal function call
    try:
        # We need to construct an AnalyzeRequest
        analyze_req = AnalyzeRequest(
            analysis_type=plan["analysis_type"],
            dataset_url=request.dataset_url,
            raw_data=request.raw_data,
            file_type=request.file_type,
            params=plan["params"]
        )
        
        # We can re-use the exact same logic from the analyze route
        from app.api.routes.analyze import run_analysis
        result = await run_analysis(analyze_req, _key)
        return result
        
    except Exception as e:
        return {
            "success": False,
            "error": f"Execution Error: {str(e)}",
            "error_code": "EXECUTION_ERROR"
        }
