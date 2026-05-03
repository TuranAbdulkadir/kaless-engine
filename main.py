"""KALESS Statistical Engine — FastAPI Application Entry Point."""

import traceback
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from config import settings

tags_metadata = [
    {"name": "Health", "description": "System health and status monitoring endpoints."},
    {"name": "Parse", "description": "Dataset ingestion and metadata extraction for .SAV, .CSV, and .XLSX files."},
    {"name": "Transform", "description": "Data manipulation operations (Recode, Compute, Sort, Merge, Weight)."},
    {"name": "Analyze", "description": "Core statistical analysis modules (T-Tests, ANOVA, Regression, Reliability)."},
    {"name": "Graphs", "description": "Visual data exploration and chart generation."},
    {"name": "Export", "description": "Report generation and data export in PDF, Word, and Excel formats."},
    {"name": "Syntax", "description": "SPSS-Compatible syntax parsing and execution engine."},
]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan — startup/shutdown events."""
    print("KALESS Engine starting up...")
    print(f"   ENV: {settings.env}")
    print(f"   CORS Origins: {settings.cors_origins}")
    print(f"   Max Upload: {settings.max_upload_size_mb} MB")
    yield
    print("KALESS Engine shutting down...")


app = FastAPI(
    title="KALESS Statistics Engine",
    description="""
    ## The High-Performance Core of the Kaless Statistics Platform.
    
    This engine provides enterprise-grade statistical processing, handling complex datasets with vectorized efficiency.
    
    ### Key Features:
    * **Legacy Compatibility**: Full support for IBM SPSS (.sav) data and syntax.
    * **Modern Stack**: Built on FastAPI, Pandas, and Statsmodels.
    * **Cloud-First**: Designed for horizontal scaling and secure session management.
    """,
    version="1.0.4",
    lifespan=lifespan,
    openapi_tags=tags_metadata,
    docs_url="/docs",
    redoc_url="/redoc",
)


# ============================================================
# MIDDLEWARE 1: Detailed Error Logging (catches ALL exceptions)
# ============================================================
@app.middleware("http")
async def error_logging_middleware(request: Request, call_next):
    start = time.time()
    try:
        response = await call_next(request)
        duration = time.time() - start
        # Log slow requests (> 5s)
        if duration > 5.0:
            print(f"⚠️ SLOW REQUEST: {request.method} {request.url.path} took {duration:.2f}s")
        return response
    except Exception as exc:
        duration = time.time() - start
        tb = traceback.format_exc()
        print(f"💥 UNHANDLED EXCEPTION on {request.method} {request.url.path}")
        print(f"   Duration: {duration:.2f}s")
        print(f"   Error: {exc}")
        print(f"   Traceback:\n{tb}")
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": f"Internal Server Error: {str(exc)}",
                "path": str(request.url.path),
            },
        )


# ============================================================
# MIDDLEWARE 2: CORS — Allow ALL origins (required for direct browser-to-engine uploads)
# ============================================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,  # Must be False when using wildcard origins
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=3600,  # Cache preflight for 1 hour
)


# ============================================================
# MIDDLEWARE 3: Request Size Limit (200MB for large .sav files)
# ============================================================
MAX_BODY_SIZE = settings.max_upload_size_mb * 1024 * 1024  # Convert to bytes


@app.middleware("http")
async def limit_upload_size(request: Request, call_next):
    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > MAX_BODY_SIZE:
        return JSONResponse(
            status_code=413,
            content={
                "success": False,
                "error": f"File too large. Maximum size is {settings.max_upload_size_mb}MB.",
            },
        )
    return await call_next(request)


# ============================================================
# ROUTE REGISTRATION
# ============================================================
from app.api.routes import analyze, graphs, export, health, parse, transform, syntax  # noqa: E402

app.include_router(health.router, prefix="/engine", tags=["Health"])
app.include_router(parse.router, prefix="/engine", tags=["Parse"])
app.include_router(transform.router, prefix="/engine", tags=["Transform"])
app.include_router(analyze.router, prefix="/engine", tags=["Analyze"])
app.include_router(graphs.router, prefix="/engine", tags=["Graphs"])
app.include_router(export.router, prefix="/engine", tags=["Export"])
app.include_router(syntax.router, prefix="/engine", tags=["Syntax"])


# ============================================================
# ROOT ENDPOINT (for Render health checks at /)
# ============================================================
@app.get("/")
async def root():
    return {"status": "ok", "service": "kaless-engine", "version": "1.0.4"}


@app.get("/api/health")
async def api_health():
    """Legacy health check path for Render."""
    return {"status": "ok", "service": "kaless-engine", "version": "1.0.4"}
