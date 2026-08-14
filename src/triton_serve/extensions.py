from functools import cache

import docker

from triton_serve.config import get_settings
from triton_serve.database import database_manager


def get_db():
    """Yields a database session safely.

    Yields:
        Session: The database session, closed when the dependency is torn down.
    """
    with database_manager.session() as session:
        yield session


@cache
def get_reconciler_docker_client() -> docker.DockerClient:
    """Long-lived Docker client for the reconciler ONLY. Short timeout so a slow daemon
    fails fast instead of blocking. Request handlers must not call Docker — read the DB."""
    return docker.from_env(timeout=get_settings().docker_timeout)


@cache
def get_builder_docker_client() -> docker.DockerClient:
    """Long-lived Docker client for the image builder ONLY. A build streams for minutes, so it
    cannot share the reconciler's fail-fast timeout."""
    return docker.from_env(timeout=get_settings().image_build_timeout)
