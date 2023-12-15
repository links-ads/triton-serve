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
    database_user: str = Field(...)
    database_pass: str = Field(...)
    database_host: str = "serve-database"
    database_port: int = 5432
    database_name: str = "triton_serve_db"

    database_default: str = "postgres"
    POSTGRES_USER: str
    POSTGRES_PASSWORD: str

    @property
    def database_url(self):
        return (
            f"postgresql://{self.database_user}:{self.database_pass}@"
            f"{self.database_host}:{self.database_port}/{self.database_name}"
        )
