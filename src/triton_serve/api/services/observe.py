from datetime import datetime, timezone

from docker import DockerClient
from docker.errors import ImageNotFound, NotFound
from docker.models.containers import Container

from triton_serve.api.services.reconcile import ObservedState
from triton_serve.database.model import Service


def _image_present(client: DockerClient, image_ref: str) -> bool:
    try:
        client.images.get(image_ref)
        return True
    except ImageNotFound:
        return False


def _uptime_seconds(state: dict) -> float | None:
    """Seconds since the container started, or None if the timestamp is missing/unparseable."""
    started_raw = state.get("StartedAt")
    if not isinstance(started_raw, str):
        return None
    try:
        started = datetime.fromisoformat(started_raw)
    except ValueError:
        return None
    if started.tzinfo is None:
        started = started.replace(tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - started).total_seconds()


def _running_fact(container: Container, boot_grace_seconds: int) -> ObservedState:
    """Health verdict for a running container. Docker owns it whenever a healthcheck is configured.

    Docker holds `starting` for the whole start period and only flips to `unhealthy` after the
    configured retries, so a second timer here could only ever fire early on a container docker
    still considers healthy. The boot grace is left to the no-healthcheck case, which has no
    other signal that the container has settled.
    """
    state = container.attrs.get("State", {})
    match (state.get("Health") or {}).get("Status"):
        case "healthy":
            return ObservedState.RUNNING
        case "unhealthy":
            return ObservedState.CRASHED
        case "starting":
            return ObservedState.BOOTING
        case _:
            uptime = _uptime_seconds(state)
            if uptime is None:
                return ObservedState.RUNNING
            return ObservedState.RUNNING if uptime >= boot_grace_seconds else ObservedState.BOOTING


def observe(client: DockerClient, service: Service, boot_grace_seconds: int) -> ObservedState:
    """Derive the current observed fact for a service from the Docker daemon (read-only).

    Args:
        client (DockerClient): The docker client.
        service (Service): The service to observe.
        boot_grace_seconds (int): How long a container without a healthcheck stays BOOTING.

    Returns:
        ObservedState: The fact the reconciler decides on.
    """
    try:
        container = client.containers.get(service.service_name)
    except NotFound:
        return ObservedState.ABSENT if _image_present(client, service.service_image) else ObservedState.IMAGE_MISSING

    if container.status == "running":
        return _running_fact(container, boot_grace_seconds)
    if container.status in ("created", "restarting"):
        return ObservedState.BOOTING
    exit_code = container.attrs.get("State", {}).get("ExitCode", 0)
    return ObservedState.EXITED_OK if exit_code == 0 else ObservedState.CRASHED
