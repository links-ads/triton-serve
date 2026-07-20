from datetime import datetime, timezone

from docker import DockerClient
from docker.errors import ImageNotFound, NotFound

from triton_serve.api.services.reconcile import ObservedFact
from triton_serve.database.model import Service


def _image_present(client: DockerClient, image_ref: str) -> bool:
    try:
        client.images.get(image_ref)
        return True
    except ImageNotFound:
        return False


def _running_fact(container, boot_grace_seconds: int) -> ObservedFact:
    state = container.attrs.get("State", {})
    health = (state.get("Health") or {}).get("Status")
    if health == "healthy":
        return ObservedFact.RUNNING
    if health in ("starting", "unhealthy"):
        return ObservedFact.BOOTING
    # no healthcheck: treat as booting until it has been up past the boot grace
    started_raw = state.get("StartedAt")
    try:
        started = datetime.fromisoformat(started_raw)
    except TypeError, ValueError:
        return ObservedFact.RUNNING
    if started.tzinfo is None:
        started = started.replace(tzinfo=timezone.utc)
    uptime = (datetime.now(timezone.utc) - started).total_seconds()
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
