"""KALESS Engine — Parse Route (dataset parsing)."""

from fastapi import APIRouter, Depends, HTTPException

from app.api.deps import verify_engine_key
from app.core.parser import parse_dataset
from app.schemas.parse import ParseRequest, ParseResponse
from app.utils.errors import ParseError
from app.utils.storage import download_file, upload_file
import io

router = APIRouter()


@router.post("/parse", response_model=ParseResponse)
async def parse_dataset_endpoint(
    request: ParseRequest,
    _key: str = Depends(verify_engine_key),
):
    """Parse an uploaded dataset file (CSV, XLSX, TSV).

    Downloads the file from Supabase Storage, parses it,
    and returns schema, preview rows, column types, and import warnings.
    """
    # Download file from Supabase Storage
    try:
        parts = request.file_url.split("/", 1)
        if len(parts) != 2:
            raise HTTPException(
                status_code=400,
                detail="Invalid file_url format. Expected: bucket/path",
            )
        bucket, path = parts
        file_content = download_file(bucket, path)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=502,
            detail=f"Failed to download file from storage: {str(e)}",
        )

    # Parse the file
    try:
        result = parse_dataset(
            file_content=file_content,
            file_type=request.file_type,
            encoding=request.encoding,
            delimiter=request.delimiter,
            max_preview_rows=request.max_preview_rows,
        )
        # Extract dataframe for parquet caching
        df = result.pop("_df", None)
        if df is not None:
            try:
                parquet_buf = io.BytesIO()
                df.to_parquet(parquet_buf, index=False)
                parquet_path = f"{path.rsplit('.', 1)[0]}_canonical.parquet"
                upload_file(bucket, parquet_path, parquet_buf.getvalue(), "application/octet-stream")
                result["parquet_path"] = parquet_path
            except Exception as e:
                print(f"Failed to generate parquet cache: {str(e)}")
                # We do not fail the parse if parquet generation fails temporarily
    except ParseError as e:
        raise HTTPException(status_code=422, detail=e.message)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected parsing error: {str(e)}",
        )

    return ParseResponse(**result)
