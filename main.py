"""KALESS Statistical Engine — FastAPI Application Entry Point."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan — startup/shutdown events."""
    # Startup
    yield
    # Shutdown


app = FastAPI(
    title="KALESS Statistical Engine",
    description="Statistical analysis engine for the KALESS platform",
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/docs" if not settings.is_production else None,
    redoc_url="/redoc" if not settings.is_production else None,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Route registration ---
from app.api.routes import analyze, chart, export, health, parse, transform  # noqa: E402

app.include_router(health.router, prefix="/engine", tags=["health"])
app.include_router(parse.router, prefix="/engine", tags=["parse"])
app.include_router(transform.router, prefix="/engine", tags=["transform"])
app.include_router(analyze.router, prefix="/engine", tags=["analyze"])
app.include_router(chart.router, prefix="/engine", tags=["chart"])
app.include_router(export.router, prefix="/engine", tags=["export"])
