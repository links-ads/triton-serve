import hashlib
import json
from dataclasses import asdict, dataclass

from triton_serve.database.model import Service

SPEC_LABEL = "triton-serve.spec"


def effective_image_ref(service: Service) -> str:
    """The image a service actually runs: its resolved row's ref, or its raw base image."""
    return service.image.image_ref if service.image is not None else service.service_image


@dataclass(frozen=True, slots=True)
class ContainerSpec:
    """Everything about a service that decides what its container is, and nothing else.

    Deployment-level inputs (the docker network, the storage repository's mounts and environment)
    are deliberately absent: they come from settings, not from the service row, and including them
    would make a settings change recreate every container at once.

    Changing these fields or the way they are serialised changes every fingerprint, which recreates
    every AVAILABLE service that is scaled up and revives FAILED ones that still have a container.
    """

    image_ref: str
    models: tuple[str, ...]
    cpu_count: int
    mem_size: int
    shm_size: int
    environment: dict[str, str]
    healthcheck: dict | None
    device_uuids: tuple[str, ...]

    @property
    def fingerprint(self) -> str:
        """A short, order-independent digest, short enough to read in `docker inspect`."""
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()[:16]


def container_spec(service: Service) -> ContainerSpec:
    """Reads a service row into the spec its container is built from.

    The single reader of these fields: the spawn path and the drift check must never disagree about
    what a service's container should be.

    Args:
        service (Service): The service to read.

    Returns:
        ContainerSpec: The spec, with relationship-ordered fields sorted so the fingerprint is stable.
    """
    resources = service.resources
    return ContainerSpec(
        image_ref=effective_image_ref(service),
        models=tuple(sorted(model.model_name for model in service.models)),
        cpu_count=resources.cpu_count,
        mem_size=resources.mem_size,
        shm_size=resources.shm_size,
        environment=dict(resources.environment_variables or {}),
        healthcheck=resources.healthcheck or None,
        # device_id is on the allocation row: same value as alloc.device.uuid, no extra query per
        # allocation on a loop that runs every tick
        device_uuids=tuple(sorted(alloc.device_id for alloc in service.device_allocations)),
    )
