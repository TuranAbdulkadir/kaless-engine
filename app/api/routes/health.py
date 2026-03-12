"""KALESS Engine — Health Check Route."""

from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
async def health_check():
    """Engine health check endpoint."""
    return {"status": "ok", "service": "kaless-engine", "version": "0.1.0"}
