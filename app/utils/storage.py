"""KALESS Engine — Supabase Storage Client."""

from supabase import create_client

from config import settings


def get_supabase_admin():
    """Get a Supabase admin client (service role key)."""
    return create_client(settings.supabase_url, settings.supabase_service_key)


def download_file(bucket: str, path: str) -> bytes:
    """Download a file from Supabase Storage."""
    client = get_supabase_admin()
    return client.storage.from_(bucket).download(path)


def upload_file(bucket: str, path: str, content: bytes, content_type: str = "application/octet-stream") -> None:
    """Upload a file to Supabase Storage."""
    client = get_supabase_admin()
    client.storage.from_(bucket).upload(
        path=path,
        file=content,
        file_options={"content-type": content_type}
    )
