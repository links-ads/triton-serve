import contextlib
import logging
import math
from typing import cast

from docker import DockerClient
from docker.errors import APIError, ImageNotFound, NotFound
from docker.models.containers import Container
from docker.models.images import Image
from docker.types import DeviceRequest
from fastapi import HTTPException
from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session

from triton_serve.api.dto import ServiceCreateBody, ServiceCreateResources, ServiceHealthcheck, ServiceUpdateBody
from triton_serve.api.models.domain import get_single_model
from triton_serve.api.services.observe import effective_image_ref
from triton_serve.builder.execute import enqueue_build
from triton_serve.builder.registry import RegistryAuth, auth_config
from triton_serve.builder.resolve import pip_dependencies, resolve_service_image
from triton_serve.config.schema import AppSettings
from triton_serve.config.traefik import TraefikConfigManager
from triton_serve.database.model import (
    APIKey,
    DesiredState,
    Device,
    DeviceAllocation,
    ImageStatus,
    KeyType,
    Model,
    RuntimeStatus,
    Service,
    ServiceResources,
    timezone_aware_now,
)

LOG = logging.getLogger("uvicorn")


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


def rebuild_service_config(
    db: Session,
    traefik: TraefikConfigManager,
    service: Service,
    service_prefix: str,
    default_keys: list[str],
) -> None:
    """Rewrites a service's Traefik config file from database truth.

    Single source of truth for a service's config: idempotent and safe to call on key
    assignment, creation, refresh, and startup sync alike. The written key set is the
    default (master) keys plus every non-expired service key associated with the service.

    Args:
        db (Session): The database session.
        traefik (TraefikConfigManager): The Traefik config manager.
        service (Service): The service whose config to rebuild.
        service_prefix (str): The url prefix to use for the service.
        default_keys (list[str]): The default/master keys always granted access.
    """
    keys = list(default_keys)
    associated_keys = (
        db.query(APIKey)
        .join(APIKey.services)
        .filter(
            Service.service_id == service.service_id,
            APIKey.key_type == KeyType.SERVICE,
            APIKey.expires_at > timezone_aware_now(),
        )
        .all()
    )
    keys.extend(api_key.value for api_key in associated_keys if api_key.value not in keys)
    traefik.add(service_prefix=service_prefix, service_name=service.service_name, api_keys=keys)


def list_services(
    db: Session,
    names: list[str] | None = None,
    runtime_statuses: list[RuntimeStatus] | None = None,
):
    """Returns all non-deleted services from the database.

    Read-only: returns the persisted runtime_status without touching Docker.

    Args:
        db (Session): The database session.
        names (list[str] | None): Optional filter on service names.
        runtime_statuses (list[RuntimeStatus] | None): Optional filter on runtime status.

    Returns:
        list[Service]: The list of services.
    """
    statement = db.query(Service).filter(Service.deleted_at.is_(None))
    if names:
        statement = statement.filter(Service.service_name.in_(names))
    if runtime_statuses:
        statement = statement.filter(Service.runtime_status.in_(runtime_statuses))
    return statement.all()


def get_service_by_id(db: Session, service_id: int) -> Service | None:
    """Returns a specific service by id, if present. Read-only, no Docker call.

    Args:
        db (Session): The database session.
        service_id (int): The id of the service.

    Returns:
        Service | None: The requested service, or None if absent.
    """
    return db.get(Service, ident=service_id)


def get_service_or_not_found(db: Session, service_id: int) -> Service:
    """Returns a live service by id, or raises 404. A deleted service counts as absent.

    Args:
        db (Session): The database session.
        service_id (int): The id of the service.

    Returns:
        Service: The requested service.

    Raises:
        HTTPException: 404 if the service does not exist or is deleted.
    """
    service = db.get(Service, ident=service_id)
    if service is None or service.deleted_at is not None:
        raise HTTPException(status_code=404, detail=f"Service with id {service_id} does not exist")
    return service


def get_service_record_by_name(db: Session, service_name: str) -> Service | None:
    """Pure DB lookup for the status projection hook. No Docker call."""
    return db.query(Service).filter(Service.service_name == service_name, Service.deleted_at.is_(None)).one_or_none()


def set_desired_state(db: Session, service_id: int, desired: DesiredState, wake: bool = False) -> None:
    service = get_service_or_not_found(db, service_id)
    service.desired_state = desired
    if wake:
        service.last_active_time = timezone_aware_now()
    db.commit()


def reset_and_wake(db: Session, service_id: int) -> None:
    service = get_service_or_not_found(db, service_id)
    service.restart_attempts = 0
    service.last_attempt_at = None
    service.last_active_time = timezone_aware_now()
    service.runtime_status = RuntimeStatus.RECOVERING
    # any managed image that is not ready is re-queued, not just a failed one: a build whose worker
    # died leaves the row BUILDING with no task behind it, and this is the only path back
    image = service.image
    if image is not None and image.managed and image.status is not ImageStatus.READY:
        image.status = ImageStatus.PENDING
        image.build_log = None
        retry_hash = image.image_hash
    else:
        retry_hash = None
    db.commit()
    if retry_hash is not None:
        enqueue_build(retry_hash)


def get_service_config(db: Session, service_id: int) -> ServiceCreateBody:
    """Returns the creation config for a service, suitable for reuse with POST /services.

    Args:
        db (Session): The database session.
        service_id (int): The id of the service.

    Returns:
        ServiceCreateBody: The service creation config.

    Raises:
        HTTPException: 404 if the service does not exist or is deleted.
    """
    service = get_service_or_not_found(db, service_id)

    allocations = service.device_allocations
    if not allocations:
        gpus = 0.0
    elif allocations[0].allocation_percentage < 100.0:
        gpus = round(allocations[0].allocation_percentage / 100, 2)
    else:
        gpus = float(len(allocations))

    res = service.resources
    return ServiceCreateBody(
        name=service.service_name,
        models=[m.model_name for m in service.models],
        docker_image=service.service_image,
        environment=res.environment_variables or {},
        timeout=service.inactivity_timeout,
        priority=service.priority,
        healthcheck=ServiceHealthcheck(**res.healthcheck) if res.healthcheck else None,
        resources=ServiceCreateResources(
            gpus=gpus,
            shm_size=res.shm_size,
            mem_size=res.mem_size,
            cpu_count=res.cpu_count,
        ),
    )


def get_available_devices(db: Session, count: int, required_percentage: float = 100.0) -> list[Device]:
    """
    Returns a list of available devices, considering the allocation percentage.

    Args:
        db (Session): The database session.
        count (int): The number of devices to return.
        required_percentage (float): The required percentage of allocation for each device.
                                     Defaults to 100.0 (full allocation).

    Returns:
        list[Device]: A list of available devices.
    """
    # Subquery to calculate the total allocation percentage for each device
    alloc_subquery = (
        select(
            DeviceAllocation.device_id,
            func.coalesce(func.sum(DeviceAllocation.allocation_percentage), 0).label("total_allocation"),
        )
        .join(Service, DeviceAllocation.service_id == Service.service_id)
        .where(Service.deleted_at.is_(None))
        .group_by(DeviceAllocation.device_id)
        .subquery()
    )

    # Main query to select available devices
    query = (
        select(Device)
        .outerjoin(alloc_subquery, Device.uuid == alloc_subquery.c.device_id)
        .where(
            or_(
                # Devices with no allocations
                alloc_subquery.c.total_allocation.is_(None),
                # Devices with enough free allocation
                (100 - alloc_subquery.c.total_allocation >= required_percentage),
            )
        )
        # Order by least allocated first
        .order_by(func.coalesce(alloc_subquery.c.total_allocation, 0))
        .limit(count)
    )

    return cast(list, db.scalars(query).all())


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
        HTTPException: 412 if the image can be neither found nor pulled.
    """
    try:
        try:
            return docker_client.images.get(image_name)
        except ImageNotFound:
            return docker_client.images.pull(image_name, auth_config=auth_config(auth))
    except APIError as e:
        if e.status_code in (401, 403):
            raise HTTPException(status_code=412, detail=f"Registry rejected credentials for {image_name}") from e
        raise HTTPException(status_code=412, detail=f"Cannot retrieve image: {e.explanation}") from e


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


def spawn_service_container(
    client: DockerClient,
    image_id: str,
    worker_name: str,
    worker_network: str,
    worker_volume: str,
    models: list[Model],
    worker_requirements: str,
    resources: ServiceCreateResources,
    devices: list | None = None,
    environment: dict[str, str] | None = None,
    healthcheck: dict | None = None,
):
    """Spawns a triton worker container.

    Args:

        client (DockerClient): The docker client.
        image_id (str): The identifier of the docker image to use.
        worker_name (str): The name of the worker container.
        worker_command (str): The command to run the docker image.
        worker_network (str): The name of the docker network to use.
        worker_volume (str): The path to the model repository, or a volume name.
        models (list[str]): The list of models to load.
        worker_requirements (str): Dependencies for the entrypoint to install at boot. Empty for a
            managed image, whose dependencies are already baked in.
        resources (ServiceCreateResources): The resources to use for the container.
        devices (list[str], optional): The list of devices to use. Defaults to None.
        environment (dict[str, str], optional): The environment variables to pass to the container. Defaults to None.
        healthcheck (dict, optional): The stored healthcheck config, or None for no healthcheck.

    Returns:
        str: The id of the created container.

    Raises:
        HTTPException: If the container could not be created.
    """
    # check if container with the same name already exists
    if worker_name in [container.name for container in client.containers.list(all=True)]:
        raise HTTPException(status_code=409, detail=f"Container with name {worker_name} already exists")

    environment = environment or {}
    environment["WORKER_REQUIREMENTS"] = worker_requirements

    # prepare the list of models to load
    triton_args = " ".join([f"--load-model={model.model_name}" for model in models])

    # prepare volumes for the container
    volumes = {str(worker_volume): {"bind": "/models", "mode": "ro"}}

    gpus, runtime = None, None
    if devices:
        runtime = "nvidia"
        gpus = [
            DeviceRequest(device_ids=[str(gpu.uuid)], capabilities=[["gpu", "nvidia", "compute"]]) for gpu in devices
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
        volumes=volumes,
        environment=environment,
        healthcheck=docker_healthcheck(healthcheck),  # type: ignore
        runtime=runtime,
        device_requests=gpus,
        nano_cpus=int(resources.cpu_count * 1e9),
        mem_limit=f"{resources.mem_size}m",
        shm_size=f"{resources.shm_size}m",
    )
    return container.id


def validate_models(db: Session, model_infos: list) -> list:
    """
    Validates the existence of specified models in the database.

    Args:
        db (Session): The database session.
        model_infos (list): List of model information to validate.

    Returns:
        list: List of validated model instances.

    Raises:
        HTTPException: If a specified model does not exist.
    """
    model_instances = []
    for model_name in model_infos:
        if model_name == "":
            raise HTTPException(status_code=422, detail="Model name cannot be empty")
        model = get_single_model(db=db, model_name=model_name)
        assert model is not None, f"Model '{model_name}' does not exist"
        model_instances.append(model)
    return model_instances


def get_allocable_devices(db: Session, required_gpus: float) -> tuple[list[Device], float]:
    """
    Retrieves available GPUs based on the required amount.

    Args:
        db (Session): The database session.
        required_gpus (float): The number of GPUs required.

    Returns:
        list: List of available GPU devices.

    Raises:
        AssertionError: If not enough GPUs are available.
    """
    if required_gpus > 0:
        # if under 1, we need to allocate a percentage of a single GPU
        if required_gpus < 1:
            gpu_count = 1
            gpu_percent = math.ceil(required_gpus * 100)
        # if over 1, we need to allocate a full GPU,
        # for simplicity we round up to the nearest integer
        else:
            gpu_count = math.ceil(required_gpus)
            gpu_percent = 100
        device_infos = get_available_devices(
            db,
            count=gpu_count,
            required_percentage=gpu_percent,
        )
        if len(device_infos) < required_gpus:
            raise AssertionError(
                f"Not enough GPUs available. Requested: {required_gpus}, Available: {len(device_infos)}"
            )
        return device_infos, gpu_percent
    return [], 0


def create_service_entry(
    db: Session,
    service_name: str,
    image_name: str,
    service_timeout: int,
    service_priority: int,
    service_resources: ServiceCreateResources,
    service_environment: dict,
    model_instances: list[Model],
    service_healthcheck: ServiceHealthcheck | None = None,
) -> Service:
    """
    Creates a new service entry in the database.

    Args:
        db (Session): The database session.
        service_name (str): The name of the service.
        image_name (str): The name of the Docker image.
        service_timeout (int): The timeout for the service.
        service_priority (int): The priority for the service.
        service_resources (ServiceResources): The resources allocated to the service.
        service_environment (dict): The environment variables for the service.
        model_instances (list): The list of model instances associated with the service.
        service_healthcheck (ServiceHealthcheck, optional): The container healthcheck, if any.

    Returns:
        Service: The created service entry.
    """
    service = Service(
        service_name=service_name,
        service_image=image_name,
        inactivity_timeout=service_timeout,
        priority=service_priority,
        created_at=timezone_aware_now(),
        last_active_time=timezone_aware_now(),
    )
    service.models.extend(model_instances)

    resources = ServiceResources(
        cpu_count=service_resources.cpu_count,
        mem_size=service_resources.mem_size,
        shm_size=service_resources.shm_size,
        environment_variables=service_environment,
        healthcheck=service_healthcheck.model_dump() if service_healthcheck else None,
    )
    service.resources = resources

    db.add(service)
    db.flush()
    return service


def create_device_allocations(
    db: Session,
    service_id: int,
    device_infos: list,
    device_percent: float,
):
    """
    Creates device allocation entries for a service.

    Args:
        db (Session): The database session.
        service_id (int): The ID of the service.
        device_infos (list): List of device information to allocate.
        device_percent (float): Allocation percentage, 100% unless partial device
    """
    for device in device_infos:
        allocation = DeviceAllocation(
            device_id=device.uuid,
            service_id=service_id,
            allocation_percentage=device_percent,
        )
        db.add(allocation)


def create_service(
    db: Session,
    traefik: TraefikConfigManager,
    settings: AppSettings,
    service_name: str,
    image_name: str,
    service_url_prefix: str,
    service_environment: dict[str, str],
    service_resources: ServiceCreateResources,
    service_timeout: int,
    service_priority: int,
    model_infos: list[str],
    service_api_keys: list[str] | None = None,
    service_healthcheck: ServiceHealthcheck | None = None,
) -> Service:
    """Declaratively creates a service record; the reconciler spawns the container out of band.

    No Docker call in the request path: the API is a desired-state store. The record persists
    with desired_state=AVAILABLE and runtime_status=WARMING, and the reconciler pulls the image
    and spawns the container on its next tick (surfacing a bad image ref as FAILED, not a 4xx here).

    Args:
        db (Session): The database session.
        traefik (TraefikConfigManager): The Traefik config manager.
        settings (AppSettings): The application settings.
        service_name (str): The name of the service.
        image_name (str): The name of the Docker image to use.
        service_url_prefix (str): The URL prefix to use for the service.
        service_environment (dict[str, str]): The environment variables to pass to the container.
        service_resources (ServiceCreateResources): The resources to use for the container.
        service_timeout (int): The timeout for the service.
        service_priority (int): The priority for the service.
        model_infos (list): The list of models to load.
        service_api_keys (list[str], optional): The list of API keys to use for the service.
        service_healthcheck (ServiceHealthcheck, optional): The container healthcheck, if any.

    Returns:
        Service: The created service.

    Raises:
        HTTPException: If capacity validation fails or the service could not be created.
    """
    try:
        assert image_name, "No image specified"
        model_instances = validate_models(db, model_infos)
        device_infos, device_percent = get_allocable_devices(db, required_gpus=service_resources.gpus)
        service = create_service_entry(
            db=db,
            service_name=service_name,
            image_name=image_name,
            service_timeout=service_timeout,
            service_priority=service_priority,
            service_resources=service_resources,
            service_environment=service_environment,
            model_instances=model_instances,
            service_healthcheck=service_healthcheck,
        )
        pending_build = resolve_service_image(db=db, service=service, settings=settings)
        create_device_allocations(
            db=db,
            service_id=service.service_id,
            device_infos=device_infos,
            device_percent=device_percent,
        )
        rebuild_service_config(
            db=db,
            traefik=traefik,
            service=service,
            service_prefix=service_url_prefix,
            default_keys=service_api_keys or [],
        )
        db.commit()
        db.refresh(service)
        # strictly after the commit: an enqueue before it could race a transaction that rolls back
        if pending_build is not None:
            enqueue_build(pending_build)
        return service

    except AssertionError as e:
        db.rollback()
        raise HTTPException(status_code=409, detail=f"Error creating service: {str(e)}") from e
    except ValueError as e:
        db.rollback()
        raise HTTPException(status_code=422, detail=f"Invalid build spec: {e}") from e
    except Exception as e:
        db.rollback()
        raise e


def delete_service(db: Session, traefik: TraefikConfigManager, service_id: int) -> None:
    """Soft-deletes a service: DB record plus synchronous Traefik teardown.

    Removes the Traefik config now (symmetric with create writing it synchronously), stamps
    deleted_at, and marks the service RETIRED. Capacity is released automatically because
    allocation/capacity queries filter deleted_at IS NULL. The reconciler removes the container
    out of band on its next tick.

    Args:
        db (Session): The database session.
        traefik (TraefikConfigManager): The Traefik config manager.
        service_id (int): The ID of the service.

    Raises:
        HTTPException: If the service does not exist or is already deleted.
    """
    service = get_service_or_not_found(db, service_id)
    traefik.delete(service_name=service.service_name)
    service.deleted_at = timezone_aware_now()
    service.desired_state = DesiredState.RETIRED
    db.commit()


def update_active_time(db: Session, service: Service):
    """Updates the last active time of a service.

    Args:
        db (Session): The database session.
        service (Service): The service to update.
    """
    service.last_active_time = timezone_aware_now()
    db.commit()


def _boot_requirements(service: Service) -> str:
    """Dependencies the entrypoint must still install at boot.

    Empty for a managed image: its dependencies are baked in, and re-installing them at boot would
    defeat the point of building it. Backfilled, unmanaged images keep today's behaviour.
    """
    if service.image is not None and service.image.managed:
        return ""
    return " ".join(pip_dependencies(service))


def recreate_service_container(
    db: Session,
    client: DockerClient,
    service: Service,
    service_network: str,
    service_models_volume: str,
    pull_credentials: RegistryAuth,
) -> Service:
    """Tears down the current container (if any) and spawns a fresh one from DB state.

    Does not touch deleted_at, Traefik config, or device allocation records.

    Args:
        db (Session): The database session.
        client (DockerClient): The Docker client.
        service (Service): The service ORM object.
        service_network (str): The Docker network name.
        service_models_volume (str): The volume name or path for models.
        pull_credentials (RegistryAuth): Credentials for pulling a private image.

    Returns:
        Service: The updated service.
    """
    try:
        if service.container_id:
            with contextlib.suppress(NotFound):
                client.containers.get(service.container_id).remove(force=True)
            service.container_id = None

        # a stale/foreign container may still hold the name under a different id (e.g. dirty
        # docker after a host reboot); clear it by name so the spawn below cannot 409.
        if (squatter := get_container_by_name(client, service.service_name)) is not None:
            squatter.remove(force=True)

        image = get_service_image(client, effective_image_ref(service), pull_credentials)
        res = service.resources
        device_objs = [alloc.device for alloc in service.device_allocations]

        container_id = spawn_service_container(
            client=client,
            image_id=cast(str, image.id),
            worker_name=service.service_name,
            worker_network=service_network,
            worker_volume=service_models_volume,
            models=service.models,
            worker_requirements=_boot_requirements(service),
            resources=ServiceCreateResources(
                gpus=0.0,
                shm_size=res.shm_size,
                mem_size=res.mem_size,
                cpu_count=res.cpu_count,
            ),
            devices=device_objs,
            environment=res.environment_variables or {},
            healthcheck=res.healthcheck,
        )

        service.container_id = str(container_id)
        db.commit()
        db.refresh(service)
        return service

    except AssertionError as e:
        db.rollback()
        raise HTTPException(status_code=409, detail=f"Error recreating service: {str(e)}") from e
    except APIError as e:
        db.rollback()
        raise HTTPException(status_code=e.status_code or 500, detail=f"Error recreating service: {str(e)}") from e


def update_service(
    db: Session,
    service_id: int,
    update_body: ServiceUpdateBody,
    settings: AppSettings,
) -> Service:
    """Applies a partial configuration change to a service record (declarative, no Docker).

    Container-affecting changes land in the record and take effect the next time the reconciler
    (re)creates the container; there is no synchronous recreate in the request path.

    Args:
        db (Session): The database session.
        service_id (int): The ID of the service to update.
        update_body (ServiceUpdateBody): The partial update payload.
        settings (AppSettings): The application settings.

    Returns:
        Service: The updated service.
    """
    try:
        service = get_service_by_id(db=db, service_id=service_id)
        if service is None:
            raise HTTPException(status_code=404, detail=f"Service with id {service_id} does not exist")
        if service.deleted_at is not None:
            raise HTTPException(status_code=409, detail="cannot update a deleted service")

        if update_body.docker_image:
            service.service_image = update_body.docker_image
        if update_body.timeout is not None:
            service.inactivity_timeout = update_body.timeout
        if update_body.priority is not None:
            service.priority = update_body.priority

        gpu_changed = False
        new_gpus = 0.0
        if update_body.resources:
            r = update_body.resources
            if r.cpu_count is not None:
                service.resources.cpu_count = r.cpu_count
            if r.shm_size is not None:
                service.resources.shm_size = r.shm_size
            if r.mem_size is not None:
                service.resources.mem_size = r.mem_size
            if r.gpus is not None:
                gpu_changed = True
                new_gpus = r.gpus

        if update_body.environment is not None:
            service.resources.environment_variables = update_body.environment

        if update_body.healthcheck is not None:
            service.resources.healthcheck = update_body.healthcheck.model_dump()

        if update_body.models is not None:
            new_model_instances = [get_single_model(db, name) for name in update_body.models]
            service.models.clear()
            service.models.extend(new_model_instances)

        if gpu_changed:
            for alloc in service.device_allocations:
                db.delete(alloc)
            db.flush()
            device_infos, device_percent = get_allocable_devices(db, required_gpus=new_gpus)
            create_device_allocations(db, service.service_id, device_infos, device_percent)

        pending_build = resolve_service_image(db=db, service=service, settings=settings)
        db.commit()
        db.refresh(service)
        if pending_build is not None:
            enqueue_build(pending_build)
        return service

    except HTTPException:
        raise
    except AssertionError as e:
        db.rollback()
        raise HTTPException(status_code=409, detail=f"Error updating service: {str(e)}") from e
    except ValueError as e:
        db.rollback()
        raise HTTPException(status_code=422, detail=f"Invalid build spec: {e}") from e
    except Exception as e:
        db.rollback()
        raise e
