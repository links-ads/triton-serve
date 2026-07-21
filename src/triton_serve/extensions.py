import docker

from triton_serve.config import get_settings
from triton_serve.database import database_manager

_reconciler_client: "docker.DockerClient | None" = None


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
