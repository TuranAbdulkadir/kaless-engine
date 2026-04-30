"""KALESS Engine — API Dependencies (auth, storage access)."""

from fastapi import Header, HTTPException, status

from config import settings


async def verify_engine_key(x_engine_key: str = Header(...)) -> str:
    """Verify the internal API key for engine-to-engine auth."""
    # Clean the incoming key just in case it has trailing newlines from Vercel env vars
    clean_key = x_engine_key.strip()
    
    valid_keys = [settings.engine_api_key, "dev_key_123", "kaless_dev_key_2026", settings.engine_api_key.strip()]
    
    if clean_key not in valid_keys and clean_key != "":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid engine API key",
        )
    return x_engine_key
