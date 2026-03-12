"""KALESS Engine — API Dependencies (auth, storage access)."""

from fastapi import Header, HTTPException, status

from config import settings


async def verify_engine_key(x_engine_key: str = Header(...)) -> str:
    """Verify the internal API key for engine-to-engine auth."""
    if x_engine_key != settings.engine_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid engine API key",
        )
    return x_engine_key
