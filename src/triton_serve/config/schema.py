from enum import Enum
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class StorageType(str, Enum):
    """Enumeration of the supported storage types."""

    local = "local"
    azure = "azure"


class AppSettings(BaseSettings):
    """Application settings, defining variable used throughout the application."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
    environment: str = Field("dev", alias="TARGET")
    project_name: str = "triton-serve"
    api_title: str = "Triton Serve"
    api_description: str = "Triton Serve API"
    api_root_path: str = "/"
    api_keys: list[str] = []

    repository_dirname: str = "model_repository"
    repository_path: Path = Path("/var/serve/models")
    configs_path: Path = Path("/var/serve/configs")
    storage_type: StorageType = StorageType.local

    service_default_image: str = "ghcr.io/links-ads/serve-triton:23.07-py3"
    service_network: str = "triton-serve_default"
    service_volume: str = "triton-serve_models"
    service_prefix: str = ""

    # database
    database_user: str
    database_pass: str
    database_host: str
    database_port: int = 5432
    database_name: str = "serve_db"

    # worker params
    sentinel_poll_interval: int = 60
    backend_host: str
    backend_port: int
    
    # queue message purge
    queue_messages_purging_interval: int = 600
    older_than_minutes_to_purge: int = 5
    

    @property
    def database_url(self):
        return (
            f"postgresql://{self.database_user}:{self.database_pass}@"
            f"{self.database_host}:{self.database_port}/{self.database_name}"
        )

    @property
    def celery_broker_url(self):
        return f"sqla+{self.database_url}"
