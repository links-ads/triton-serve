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

    project_name: str = "triton-serve"
    api_title: str = "Triton Serve"
    api_description: str = "Triton Serve API"
    api_root_path: str = "/"

    repository_path: Path = Path("/var/serve/models")
    configs_path: Path = Path("/var/serve/configs")
    storage_type: StorageType = StorageType.local

    service_image: str = "nvcr.io/nvidia/tritonserver:23.07-py3"
    service_command: str = "tritonserver --model-repository=/models --model-control-mode=explicit"
    service_network: str | None = None
    service_volume: str = "triton-serve_models"
    service_prefix: str = ""

    # security
    app_secret: str = Field(...)

    # database
    db_user: str = Field(...)
    db_password: str = Field(...)
    db_host: str = "serve-database"
    db_port: int = 5432
    db_name: str = "devices_db"
