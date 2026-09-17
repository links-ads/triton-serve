from enum import StrEnum
from pathlib import Path

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class StorageType(StrEnum):
    """Enumeration of the supported storage types."""

    local = "local"
    azure = "azure"


# seconds the hard time limit sits above the soft one: room for the task to record the failure
# before it is killed, and the margin the broker band below has to clear
BUILD_HARD_LIMIT_MARGIN = 60


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
    image_build_stale_after: int = 3600  # seconds a PENDING/BUILDING row may sit before it is reaped

    # database
    database_user: str
    database_pass: str
    database_host: str
    database_port: int = 5432
    database_name: str = "serve_db"

    # broker
    redis_host: str = "redis"
    redis_port: int = 6379

    # worker params
    sentinel_poll_interval: int = 10  # reconcile tick; the loop is a cheap DB read + docker inspect
    docker_timeout: int = 10  # seconds; reconciler Docker client, fail fast not 60s
    service_boot_grace: int = 30  # seconds a no-healthcheck container is BOOTING
    backend_host: str
    backend_port: int

    @property
    def database_url(self):
        return (
            f"postgresql://{self.database_user}:{self.database_pass}@"
            f"{self.database_host}:{self.database_port}/{self.database_name}"
        )

    @property
    def celery_broker_url(self) -> str:
        return f"redis://{self.redis_host}:{self.redis_port}/0"

    @property
    def broker_visibility_timeout(self) -> int:
        """Seconds a delivered message stays invisible before Redis hands it to another worker.

        The midpoint of the band the build path requires: above one attempt, so a healthy build is
        never redelivered while the first worker still runs it, and below the reaper's threshold,
        so a lost attempt comes back before the row is failed. Redis defaults this to exactly
        `image_build_stale_after`, which races the reaper, hence pinning it.
        """
        return self.image_build_timeout + (self.image_build_stale_after - self.image_build_timeout) // 2

    @property
    def image_build_hard_limit(self) -> int:
        """Seconds after which a build attempt is killed outright, backstopping the soft limit.

        A task that ignores the soft signal still has to die before the broker redelivers its
        message, or a second replica can start the same build while the first one still runs.
        """
        return self.image_build_timeout + BUILD_HARD_LIMIT_MARGIN

    @model_validator(mode="after")
    def _check_build_bounds(self) -> AppSettings:
        if self.image_build_timeout >= self.image_build_stale_after:
            raise ValueError(
                "image_build_timeout must be smaller than image_build_stale_after: the broker's "
                "visibility timeout has to fit between them"
            )
        if self.image_build_hard_limit >= self.broker_visibility_timeout:
            raise ValueError(
                "image_build_stale_after must exceed image_build_timeout by more than "
                f"{2 * BUILD_HARD_LIMIT_MARGIN} seconds, so the attempt is killed before the "
                "broker's visibility timeout redelivers its message"
            )
        return self
