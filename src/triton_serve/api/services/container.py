import contextlib
from typing import cast

from docker import DockerClient
from docker.errors import APIError, ImageNotFound, NotFound
from docker.models.containers import Container
from docker.models.images import Image
from docker.types import DeviceRequest
from sqlalchemy.orm import Session

from triton_serve.api.services.spec import SPEC_LABEL, ContainerSpec, container_spec
from triton_serve.builder.registry import RegistryAuth, auth_config
from triton_serve.database.model import Service
from triton_serve.storage import WorkerRepository


class ContainerError(RuntimeError):
    """A container or image operation failed on the daemon.

    The reconciler is the only caller of this module, and it has no HTTP client waiting on it, so
    failures surface as a plain error rather than as an `HTTPException` nobody is listening for.
    """


def get_container_by_name(client: DockerClient, name: str) -> Container | None:
    """Returns the container currently holding `name` (in any state), or None if absent.

    Lookup is by name rather than id: a MISSING service's stored container_id is exactly what
    no longer resolves, while a container under the service name may still exist (e.g. it came
    back under a new id after a host reboot, or a stale one is squatting the name).
    """
    try:
        return client.containers.get(name)
    except NotFound:
        return None


def get_service_image(docker_client: DockerClient, image_name: str, auth: RegistryAuth) -> Image:
    """Returns a local image, pulling it from the registry if it is not present.

    Images are private, so the pull is always authenticated when credentials are configured. An
    unauthenticated pull of a private package 404s, which would otherwise surface as a missing
    image rather than as the auth error it is.

    Args:
        docker_client (DockerClient): The docker client.
        image_name (str): The full reference of the image.
        auth (RegistryAuth): The credential provider for the pull.

    Returns:
        Image: The local image.

    Raises:
        ContainerError: If the image can be neither found nor pulled.
    """
    try:
        try:
            return docker_client.images.get(image_name)
        except ImageNotFound:
            return docker_client.images.pull(image_name, auth_config=auth_config(auth))
    except APIError as e:
        if e.status_code in (401, 403):
            raise ContainerError(f"Registry rejected credentials for {image_name}") from e
        raise ContainerError(f"Cannot retrieve image: {e.explanation}") from e


def docker_healthcheck(healthcheck: dict | None) -> dict | None:
    """Converts a stored healthcheck (seconds, snake_case) to the docker API shape (ns, PascalCase).

    Services store the user-facing shape so it round-trips through the API unchanged; docker only
    accepts durations in nanoseconds. Returns None when the service has no healthcheck configured,
    which leaves the container without one and falls back to the boot-grace timer in `observe`.
    """
    if not healthcheck:
        return None
    return {
        "Test": healthcheck["test"],
        "Interval": int(healthcheck["interval"] * 1e9),
        "Timeout": int(healthcheck["timeout"] * 1e9),
        "Retries": healthcheck["retries"],
        "StartPeriod": int(healthcheck["start_period"] * 1e9),
    }


def merge_environment(user: dict[str, str], storage: dict[str, str]) -> dict[str, str]:
    """Storage wiring wins: a service creator must not be able to repoint a worker's repository."""
    return {**user, **storage}


def spawn_service_container(
    client: DockerClient,
    image_id: str,
    worker_name: str,
    worker_network: str,
    repository: WorkerRepository,
    spec: ContainerSpec,
) -> str:
    """Spawns a triton worker container built from `spec`, labelled with its fingerprint.

    Args:
        client (DockerClient): The docker client.
        image_id (str): The identifier of the docker image to use.
        worker_name (str): The name of the worker container.
        worker_network (str): The name of the docker network to use.
        repository (WorkerRepository): The mounts and environment that point the worker at the
            model repository.
        spec (ContainerSpec): The service-level inputs the container is built from, devices included.

    Returns:
        str: The id of the created container.

    Raises:
        ContainerError: If a container already holds the name.
    """
    if worker_name in [container.name for container in client.containers.list(all=True)]:
        raise ContainerError(f"Container with name {worker_name} already exists")

    environment = merge_environment(spec.environment, repository.environment)
    triton_args = " ".join(f"--load-model={name}" for name in spec.models)

    gpus, runtime = None, None
    if spec.device_uuids:
        runtime = "nvidia"
        gpus = [
            DeviceRequest(device_ids=[uuid], capabilities=[["gpu", "nvidia", "compute"]]) for uuid in spec.device_uuids
        ]

    # no restart_policy: the reconciler owns restarts. a docker-level on-failure policy would
    # restart the container behind its back, showing up as `restarting` (-> BOOTING) and silently
    # multiplying the crash budget by the policy's retry count.
    container = client.containers.run(
        detach=True,
        remove=False,
        image=image_id,
        name=worker_name,
        command=triton_args,
        network=worker_network,
        volumes=repository.mounts,
        environment=environment,
        healthcheck=docker_healthcheck(spec.healthcheck),  # type: ignore
        labels={SPEC_LABEL: spec.fingerprint},
        runtime=runtime,
        device_requests=gpus,
        nano_cpus=int(spec.cpu_count * 1e9),
        mem_limit=f"{spec.mem_size}m",
        shm_size=f"{spec.shm_size}m",
    )
    return str(container.id)


def recreate_service_container(
    db: Session,
    client: DockerClient,
    service: Service,
    service_network: str,
    repository: WorkerRepository,
    pull_credentials: RegistryAuth,
) -> Service:
    """Tears down the current container (if any) and spawns a fresh one from DB state.

    Does not touch deleted_at, Traefik config, or device allocation records.

    Args:
        db (Session): The database session.
        client (DockerClient): The Docker client.
        service (Service): The service ORM object.
        service_network (str): The Docker network name.
        repository (WorkerRepository): The mounts and environment that point the worker at the
            model repository.
        pull_credentials (RegistryAuth): Credentials for pulling a private image.

    Returns:
        Service: The updated service.

    Raises:
        ContainerError: If the container could not be recreated.
    """
    try:
        if service.container_id:
            with contextlib.suppress(NotFound):
                client.containers.get(service.container_id).remove(force=True)
            service.container_id = None

        # a stale/foreign container may still hold the name under a different id (e.g. dirty
        # docker after a host reboot); clear it by name so the spawn below cannot clash.
        if (squatter := get_container_by_name(client, service.service_name)) is not None:
            squatter.remove(force=True)

        spec = container_spec(service)
        image = get_service_image(client, spec.image_ref, pull_credentials)
        container_id = spawn_service_container(
            client=client,
            image_id=cast(str, image.id),
            worker_name=service.service_name,
            worker_network=service_network,
            repository=repository,
            spec=spec,
        )

        service.container_id = str(container_id)
        db.commit()
        db.refresh(service)
        return service

    except (AssertionError, APIError) as e:
        db.rollback()
        raise ContainerError(f"Error recreating service: {e!s}") from e
