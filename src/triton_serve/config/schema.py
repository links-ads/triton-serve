from enum import StrEnum
from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class StorageType(StrEnum):
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
    service_max_restart_attempts: int = 3
    service_restart_cooldown: int = 600  # seconds; also the READY window that earns the budget back
    service_restart_backoff_base: int = 10  # seconds; base of the exponential retry backoff

    # image registry; tokens are never logged and never written into a build context
    registry_url: str = "ghcr.io"
    registry_namespace: str = "links-ads"
    registry_image_name: str = "serve-runtime"
    registry_push_username: str = ""
    registry_push_token: SecretStr = SecretStr("")
    registry_pull_username: str = ""
    registry_pull_token: SecretStr = SecretStr("")
    image_build_timeout: int = 1800  # seconds; a build streams for minutes, unlike a reconcile call

    # database
    database_user: str
    database_pass: str
    database_host: str
    database_port: int = 5432
    database_name: str = "serve_db"

    # worker params
    sentinel_poll_interval: int = 10  # reconcile tick; the loop is a cheap DB read + docker inspect
    docker_timeout: int = 10  # seconds; reconciler Docker client, fail fast not 60s
    service_boot_grace: int = 30  # seconds a no-healthcheck container is BOOTING
    backend_host: str
    backend_port: int

    # queue message purge
    purge_message_schedule: int = 86400
    older_than_hours_to_purge: int = 24

    @property
    def database_url(self):
        return (
            f"postgresql://{self.database_user}:{self.database_pass}@"
            f"{self.database_host}:{self.database_port}/{self.database_name}"
        )

    @property
    def celery_broker_url(self):
        return f"sqla+{self.database_url}"
