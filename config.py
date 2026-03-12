"""KALESS Statistical Engine — Configuration."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Supabase
    supabase_url: str = ""
    supabase_service_key: str = ""

    # Engine Auth
    engine_api_key: str = ""

    # Redis
    redis_url: str = "redis://localhost:6379/0"

    # CORS
    allowed_origins: str = "http://localhost:3000"

    # App
    env: str = "development"
    log_level: str = "info"
    max_upload_size_mb: int = 200

    @property
    def is_production(self) -> bool:
        return self.env == "production"

    @property
    def cors_origins(self) -> list[str]:
        return [o.strip() for o in self.allowed_origins.split(",")]

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
