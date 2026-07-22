from datetime import datetime, timezone

from docker import DockerClient
from docker.errors import ImageNotFound, NotFound
from docker.models.containers import Container

from triton_serve.api.services.reconcile import ObservedFact
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


def _running_fact(container: Container, boot_grace_seconds: int) -> ObservedFact:
    state = container.attrs.get("State", {})
    health = (state.get("Health") or {}).get("Status")
    if health == "healthy":
        return ObservedFact.RUNNING
    uptime = _uptime_seconds(state)
    if health in ("starting", "unhealthy"):
        # still coming up within the grace window; a healthcheck still failing past it is a crash
        if uptime is None:
            return ObservedFact.BOOTING
        return ObservedFact.BOOTING if uptime < boot_grace_seconds else ObservedFact.CRASHED
    # no healthcheck: treat as booting until it has been up past the boot grace
    if uptime is None:
        return ObservedFact.RUNNING
    return ObservedFact.RUNNING if uptime >= boot_grace_seconds else ObservedFact.BOOTING


def observe(client: DockerClient, service: Service, boot_grace_seconds: int) -> ObservedFact:
    """Derive the current observed fact for a service from the Docker daemon (read-only)."""
    try:
        container = client.containers.get(service.service_name)
    except NotFound:
        return ObservedFact.ABSENT if _image_present(client, service.service_image) else ObservedFact.IMAGE_MISSING

    if container.status == "running":
        return _running_fact(container, boot_grace_seconds)
    if container.status in ("created", "restarting"):
        return ObservedFact.BOOTING
    exit_code = container.attrs.get("State", {}).get("ExitCode", 0)
    return ObservedFact.EXITED_OK if exit_code == 0 else ObservedFact.CRASHED
