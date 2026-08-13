import docker

from triton_serve.config import get_settings
from triton_serve.database import database_manager

_reconciler_client: "docker.DockerClient | None" = None
_builder_client: "docker.DockerClient | None" = None


def get_db():
    """Yields a database session safely.

    :yield: database session
    :rtype: Iterator[Session]
    """
    with database_manager.session() as session:
        yield session


def get_reconciler_docker_client() -> docker.DockerClient:
    """Long-lived Docker client for the reconciler ONLY. Short timeout so a slow daemon
    fails fast instead of blocking. Request handlers must not call Docker — read the DB."""
    global _reconciler_client
    if _reconciler_client is None:
        settings = get_settings()
        _reconciler_client = docker.from_env(timeout=settings.docker_timeout)
    return _reconciler_client


def get_builder_docker_client() -> docker.DockerClient:
    """Long-lived Docker client for the image builder ONLY. A build streams for minutes, so it
    cannot share the reconciler's fail-fast timeout."""
    global _builder_client
    if _builder_client is None:
        settings = get_settings()
        _builder_client = docker.from_env(timeout=settings.image_build_timeout)
    return _builder_client
