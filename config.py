"""KALESS Statistical Engine — Configuration."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Supabase
    supabase_url: str = "https://kczykrrhtdsjtbqiemvph.supabase.co"
    supabase_service_key: str = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Imtjenlrcmh0ZHNqdGJxaWVtdnBoIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc3MzMwNzc4MSwiZXhwIjoyMDg4ODgzNzgxfQ.nOkuYcIQu6SAxDoonqa_ZdoOGuO-seoqX_4blcjO3ow"

    # Engine Auth
    engine_api_key: str = ""

    # Redis
    redis_url: str = "redis://localhost:6379/0"

    # CORS — comma-separated list of allowed origins
    allowed_origins: str = (
        "http://localhost:3000,"
        "http://localhost:3001,"
        "https://kaless-web.vercel.app,"
        "https://kaless-web-turanabdulkadirs-projects.vercel.app,"
        "https://kaless-1os388mys-turanabdulkadirs-projects.vercel.app"
    )

    # App
    env: str = "development"
    log_level: str = "info"
    max_upload_size_mb: int = 200

    @property
    def is_production(self) -> bool:
        return self.env == "production"

    @property
    def cors_origins(self) -> list[str]:
        origins = [o.strip() for o in self.allowed_origins.split(",") if o.strip()]
        # Always ensure Vercel preview URLs are allowed
        origins.append("https://kaless-web.vercel.app")
        return list(set(origins))  # deduplicate

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
