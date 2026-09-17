import signal
from datetime import UTC, datetime

from docker import DockerClient
from docker.errors import ImageNotFound, NotFound
from docker.models.containers import Container

from triton_serve.api.services.reconcile import ObservedState
from triton_serve.database.model import ImageStatus, Service, timezone_aware_now

# 128+N is the exit status of a process killed by signal N. Docker's stop sends SIGTERM and then
# SIGKILL once the grace expires, so a container the reconciler stopped itself reports one of these
# rather than having failed on its own.
_STOPPED_BY_SIGNAL = frozenset({128 + signal.SIGTERM, 128 + signal.SIGKILL})


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
        started = started.replace(tzinfo=UTC)
    return (timezone_aware_now() - started).total_seconds()


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


def effective_image_ref(service: Service) -> str:
    """The image a service actually runs: its resolved row's ref, or its raw base image."""
    return service.image.image_ref if service.image is not None else service.service_image


def observe(
    client: DockerClient,
    service: Service,
    boot_grace_seconds: int,
    image_status: ImageStatus | None,
) -> ObservedState:
    """Derive the current observed fact for a service from its image row and the Docker daemon.

    Read-only. The image status is checked first: there is nothing useful to observe on the
    daemon for a service whose image does not exist yet.

    Args:
        client (DockerClient): The docker client.
        service (Service): The service to observe.
        boot_grace_seconds (int): How long a container without a healthcheck stays BOOTING.
        image_status (ImageStatus | None): The status of the service's resolved image row.

    Returns:
        ObservedState: The fact the reconciler decides on.
    """
    if image_status in (ImageStatus.PENDING, ImageStatus.BUILDING):
        return ObservedState.IMAGE_PENDING
    if image_status is ImageStatus.FAILED:
        return ObservedState.IMAGE_FAILED

    try:
        container = client.containers.get(service.service_name)
    except NotFound:
        present = _image_present(client, effective_image_ref(service))
        return ObservedState.ABSENT if present else ObservedState.IMAGE_MISSING

    if container.status == "running":
        return _running_fact(container, boot_grace_seconds)
    if container.status in ("created", "restarting"):
        return ObservedState.BOOTING
    state = container.attrs.get("State", {})
    exit_code = state.get("ExitCode", 0)
    # the kernel's OOM killer uses SIGKILL too, and that one is a genuine crash to back off from
    stopped_by_signal = exit_code in _STOPPED_BY_SIGNAL and not state.get("OOMKilled")
    return ObservedState.EXITED_OK if exit_code == 0 or stopped_by_signal else ObservedState.CRASHED
