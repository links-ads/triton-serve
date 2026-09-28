import logging
import math
from typing import cast

from fastapi import HTTPException
from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session

from triton_serve.api.dto import ServiceCreateBody, ServiceCreateResources, ServiceHealthcheck, ServiceUpdateBody
from triton_serve.api.models.domain import get_single_model
from triton_serve.builder.execute import enqueue_build
from triton_serve.builder.resolve import resolve_service_image
from triton_serve.config.schema import AppSettings
from triton_serve.config.traefik import TraefikConfigManager
from triton_serve.database.model import (
    APIKey,
    DesiredState,
    Device,
    DeviceAllocation,
    ImageStatus,
    Model,
    RuntimeStatus,
    Service,
    ServiceResources,
    key_service_association,
    timezone_aware_now,
)

LOG = logging.getLogger("uvicorn")


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


def get_service_with_key_association(db: Session, service_name: str, key: APIKey) -> tuple[Service, bool] | None:
    """Resolves a service by name together with whether one key is associated with it.

    One statement rather than two: testing membership by walking `key.services` hydrates every
    service the key can reach, which grows with the key rather than with the question being asked.

    Args:
        db (Session): The database session.
        service_name (str): The name of the service being requested.
        key (APIKey): The authenticated key.

    Returns:
        tuple[Service, bool] | None: The service and whether the key is associated with it, or
            None when no live service carries the name.
    """
    associated = (
        select(key_service_association.c.service_id)
        .where(
            key_service_association.c.api_key_id == key.key_id,
            key_service_association.c.service_id == Service.service_id,
        )
        .exists()
    )
    row = (
        db.query(Service, associated.label("associated"))
        .filter(Service.service_name == service_name, Service.deleted_at.is_(None))
        .one_or_none()
    )
    return (row.Service, row.associated) if row is not None else None


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
        # restamped because the row may already be PENDING: the stamp is what the reaper reads, and
        # without restarting it the sweep can fail the row before the builder picks it up
        image.transition(ImageStatus.PENDING, restamp=True)
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
        tuple[list[Device], float]: The devices to allocate, and the percentage to take of each.

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
        traefik.add(service_prefix=service_url_prefix, service_name=service.service_name)
        db.commit()
        db.refresh(service)
        # strictly after the commit: an enqueue before it could race a transaction that rolls back
        if pending_build is not None:
            enqueue_build(pending_build)
        return service

    except AssertionError as e:
        db.rollback()
        raise HTTPException(status_code=409, detail=f"Error creating service: {e!s}") from e
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


# suppressed writes age last_active_time, which the reconciler compares against inactivity_timeout:
# too much staleness would scale a service to zero under traffic, so a hundredth keeps the margin
# wide, and integer division makes any timeout under 100s write every time.
LIVENESS_WRITE_DIVISOR = 100


def record_activity(db: Session, service: Service) -> None:
    """Records that a service is being used, coalescing writes the reconciler cannot observe.

    Args:
        db (Session): The database session.
        service (Service): The service being reached.
    """
    now = timezone_aware_now()
    liveness_write_window = service.inactivity_timeout // LIVENESS_WRITE_DIVISOR
    if (now - service.last_active_time).total_seconds() < liveness_write_window:
        return
    service.last_active_time = now
    db.commit()


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
        raise HTTPException(status_code=409, detail=f"Error updating service: {e!s}") from e
    except ValueError as e:
        db.rollback()
        raise HTTPException(status_code=422, detail=f"Invalid build spec: {e}") from e
    except Exception as e:
        db.rollback()
        raise e
