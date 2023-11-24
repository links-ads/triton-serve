from enum import Enum
from pathlib import Path
from pydantic import ConfigDict

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
    )

    title: str = "Triton Serve"
    description: str = "Triton Serve API"

    project_name: str = "triton-serve"

    repository_path: Path = Path("/var/serve/models")
    configs_path: Path = Path("/var/serve/configs")
    storage_type: StorageType = StorageType.local

    service_image: str = "nvcr.io/nvidia/tritonserver:23.07-py3"
    service_command: str = "tritonserver --model-repository=/models --model-control-mode=explicit"
    service_network: str | None = None
    service_volume: str = "triton-serve_models"
    service_prefix: str = ""

    # test settings
    zips_path: Path = Path("/var/serve/zips")
    tests_dir: Path = Path("./tests")
