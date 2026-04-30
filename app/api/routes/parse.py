"""KALESS Engine — Parse Route (dataset parsing)."""

import io
import traceback
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from typing import Optional

from app.api.deps import verify_engine_key
from app.core.parser import parse_dataset
from app.schemas.parse import ParseRequest, ParseResponse
from app.utils.errors import ParseError
from app.utils.storage import download_file, upload_file

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
    except ParseError as e:
        raise HTTPException(status_code=422, detail=e.message)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected parsing error: {str(e)}",
        )

    return ParseResponse(**result)


@router.post("/parse/upload")
async def parse_upload_direct(
    file: UploadFile = File(...),
    project_id: Optional[str] = Form(None),
):
    """Direct file upload + parse endpoint.
    
    Accepts multipart file upload (CSV, XLSX, SAV),
    parses it in-memory with pandas/pyreadstat,
    and returns structured JSON with dataset info, variables, and preview rows.
    
    No Supabase storage required — pure server-side processing.
    """
    print(f"📥 UPLOAD RECEIVED: {file.filename} ({file.content_type})")
    
    try:
        content = await file.read()
        filename = file.filename or "unknown.csv"
        ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else "csv"
        
        print(f"   File size: {len(content)} bytes, Extension: .{ext}")
        
        import pandas as pd
        import numpy as np
        
        df = None
        meta = {}
        
        # ── CSV ──
        if ext == "csv":
            text = content.decode("utf-8", errors="replace")
            df = pd.read_csv(io.StringIO(text))
            
        # ── TSV ──
        elif ext == "tsv":
            text = content.decode("utf-8", errors="replace")
            df = pd.read_csv(io.StringIO(text), sep="\t")
            
        # ── XLSX / XLS ──
        elif ext in ("xlsx", "xls"):
            df = pd.read_excel(io.BytesIO(content))
            
        # ── SAV (SPSS) ──
        elif ext == "sav":
            try:
                import pyreadstat
                # BULLETPROOF SAV PARSING: Encoding fallbacks for SPSS files with Turkish/special chars
                df = None
                meta_obj = None
                encodings = ['UTF-8', 'cp1254', 'latin1']
                
                for enc in encodings:
                    try:
                        df, meta_obj = pyreadstat.read_sav(io.BytesIO(content), encoding=enc)
                        print(f"   SAV successfully parsed with encoding: {enc}")
                        break
                    except Exception as enc_err:
                        print(f"   [Warning] SAV parse failed with encoding {enc}: {enc_err}")
                        continue
                
                if df is None or meta_obj is None:
                    # Final fallback without forcing encoding (let pyreadstat guess)
                    df, meta_obj = pyreadstat.read_sav(io.BytesIO(content))
                    print("   SAV successfully parsed with pyreadstat default auto-encoding")

                meta = {
                    "column_names_to_labels": meta_obj.column_names_to_labels or {},
                    "value_labels": meta_obj.value_labels or {},
                    "variable_measure": getattr(meta_obj, "variable_measure", {}),
                }
                
                # FATAL ERROR AVOIDANCE: Force standard types to prevent JSON serialization crash
                # Convert SPSS categorical/factor types to standard strings, numerics stay numeric
                for col in df.columns:
                    if pd.api.types.is_categorical_dtype(df[col]) or df[col].dtype == object:
                        df[col] = df[col].astype(str)
                df = df.replace(["nan", "None", "<NA>"], None)

                print(f"   SAV parsed: {len(df)} rows, {len(df.columns)} columns")
            except ImportError:
                raise HTTPException(
                    status_code=500,
                    detail="pyreadstat not installed on server. Cannot parse .sav files."
                )
            except Exception as e:
                print(f"   SAV parse error: {traceback.format_exc()}")
                raise HTTPException(
                    status_code=422,
                    detail=f"Failed to parse .sav file: {str(e)}"
                )
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file format: .{ext}"
            )
        
        if df is None or df.empty:
            raise HTTPException(status_code=422, detail="File is empty or could not be parsed.")
        
        # Replace NaN/Inf with None for JSON serialization
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.where(pd.notnull(df), None)
        
        # Build variables list
        column_labels = meta.get("column_names_to_labels", {})
        variable_measure = meta.get("variable_measure", {})
        
        variables = []
        for idx, col in enumerate(df.columns):
            is_numeric = pd.api.types.is_numeric_dtype(df[col])
            measure = variable_measure.get(col, "scale" if is_numeric else "nominal")
            variables.append({
                "id": f"v{idx}",
                "name": str(col),
                "label": column_labels.get(col, ""),
                "data_type": "numeric" if is_numeric else "string",
                "measure_level": measure,
                "column_index": idx,
                "role": "input",
                "missing_values": None,
                "notes": None,
            })
        
        # Preview rows (first 500)
        preview_rows = df.head(500).to_dict(orient="records")
        
        # Convert numpy types to native Python types for JSON
        def convert_value(v):
            if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
                return None
            if isinstance(v, (np.integer,)):
                return int(v)
            if isinstance(v, (np.floating,)):
                return float(v)
            if isinstance(v, (np.bool_,)):
                return bool(v)
            return v
        
        preview_rows = [
            {k: convert_value(v) for k, v in row.items()}
            for row in preview_rows
        ]
        
        result = {
            "success": True,
            "dataset": {
                "id": project_id or "auto",
                "name": filename,
                "file_type": ext,
                "row_count": len(df),
                "column_count": len(df.columns),
            },
            "variables": variables,
            "preview_rows": preview_rows,
        }
        
        print(f"✅ PARSE SUCCESS: {len(df)} rows × {len(df.columns)} cols")
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        tb = traceback.format_exc()
        print(f"💥 UPLOAD PARSE FAILED: {e}")
        print(f"   Traceback:\n{tb}")
        raise HTTPException(
            status_code=500,
            detail=f"Upload processing failed: {str(e)}"
        )
