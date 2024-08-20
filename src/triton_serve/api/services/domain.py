import logging
from datetime import datetime, timezone
from itertools import chain
from pathlib import Path

from docker import DockerClient
from docker.models.images import Image
from docker.errors import APIError, ImageNotFound, NotFound, NullResource
from docker.types import DeviceRequest
from fastapi import HTTPException
from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session

from triton_serve.api.dto import ServiceCreateResources
from triton_serve.api.models.domain import get_single_model
from triton_serve.config.traefik import TraefikConfigManager
from triton_serve.database.model import (
    Device,
    DeviceAllocation,
    Model,
    Service,
    ServiceResources,
    ServiceStatus,
)
from triton_serve.storage import ModelStorage

LOG = logging.getLogger("uvicorn")


def list_services(
    db: Session,
    docker_client: DockerClient,
    names: list[str] = None,
    statuses: list[ServiceStatus] = None,
):
    """Returns a list of all services.

    Args:
        db (Session): The database session.

    Returns:
        list[Service]: The list of services.
    """
    statement = db.query(Service)
    if names:
        statement = statement.filter(Service.service_name.in_(names))
    if statuses:
        statement = statement.filter(Service.container_status.in_(statuses))
    services = statement.all()
    return [check_service_status(db=db, docker_client=docker_client, service=service) for service in services]


def get_service_by_id(db: Session, service_id: int, docker_client: DockerClient | None = None):
    """Returns a specific service by id, if present.

    Args:
        db (Session): The database session.
        service_id (int): The id of the service.

    Returns:
        Service: The requested service.
    """
    service = db.get(Service, ident=service_id)
    if docker_client is not None:
        return check_service_status(db=db, docker_client=docker_client, service=service)
    return service


def get_service_by_name(db: Session, service_name: str, docker_client: DockerClient | None = None):
    """Returns a specific service by name, if present.

    Args:
        db (Session): The database session.
        service_name (str): The name of the service.

    Returns:
        Service: The requested service.
    """
    # get any service with the specified name, not deleted
    service = (
        db.query(Service)
        .filter(
            Service.service_name == service_name,
            Service.deleted_at.is_(None),
        )
        .first()
    )
    if service is None:
        raise HTTPException(status_code=404, detail=f"No active service named '{service_name}'")
    if docker_client is not None:
        return check_service_status(db=db, docker_client=docker_client, service=service)
    return service


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

    return db.scalars(query).all()


def get_service_image(docker_client: DockerClient, image_name: str) -> Image:
    """Returns the image id of a docker image.
    If it does not exist, it tries to pull it from the registry, otherwise raises
    an HTTPException with code 412, and the message "Image not found".

    Args:
        docker_client (DockerClient): The docker client.
        image_name (str): The name of the image.

    Returns:
        str: The image id.
    """
    try:
        try:
            image = docker_client.images.get(image_name)
            return image
        except ImageNotFound:
            image = docker_client.images.pull(image_name)
            return image
    except APIError as e:
        raise HTTPException(status_code=412, detail=f"Cannot retrieve image: {e.explanation}") from e


def check_service_status(db: Session, docker_client: DockerClient, service: Service):
    """Checks the status of a service, querying the docker daemon, and updating the database.

    Args:
        db (Session): The database session.
        docker_client (DockerClient): The docker client.
        service (Service): The service to check.
    """
    try:
        container = docker_client.containers.get(service.container_id)
        if container.status == "exited" and service.container_status not in (
            ServiceStatus.STOPPED,
            ServiceStatus.ERROR,
        ):
            state = docker_client.api.inspect_container(service.container_id).get("State")
            exit_code = state.get("ExitCode", 0)
            service.container_status = ServiceStatus.STOPPED if exit_code == 0 else ServiceStatus.ERROR
        elif container.status == "running" and service.container_status != ServiceStatus.ACTIVE:
            service.container_status = ServiceStatus.ACTIVE
        db.commit()
        db.refresh(service)
        return service
    except (NotFound, NullResource):
        if service.deleted_at is None:
            service.deleted_at = datetime.now(tz=timezone.utc)
        service.container_status = ServiceStatus.DELETED
        db.commit()
        db.refresh(service)
        return service
    except APIError as e:
        service.status = ServiceStatus.ERROR
        db.commit()
        raise HTTPException(status_code=e.status_code, detail=str(e)) from e


def spawn_service_container(
    client: DockerClient,
    image_id: str,
    worker_name: str,
    worker_network: str,
    worker_volume: str,
    models: list[Model],
    resources: ServiceCreateResources,
    devices: list = None,
    environment: dict[str, str] = None,
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
        resources (ServiceCreateResources): The resources to use for the container.
        devices (list[str], optional): The list of devices to use. Defaults to None.
        environment (dict[str, str], optional): The environment variables to pass to the container. Defaults to None.

    Returns:
        str: The id of the created container.

    Raises:
        HTTPException: If the container could not be created.
    """
    # check if container with the same name already exists
    if worker_name in [container.name for container in client.containers.list()]:
        raise HTTPException(status_code=409, detail=f"Container with name {worker_name} already exists")

    # prepare the requirements, if any
    environment = environment or {}
    dependencies_list = set(chain.from_iterable([model.dependencies for model in models]))
    environment["WORKER_REQUIREMENTS"] = " ".join(dependencies_list) if dependencies_list else ""

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

    container = client.containers.run(
        detach=True,
        remove=False,
        image=image_id,
        name=worker_name,
        command=triton_args,
        network=worker_network,
        volumes=volumes,
        environment=environment,
        restart_policy={"Name": "unless-stopped"},
        runtime=runtime,
        device_requests=gpus,
        nano_cpus=int(resources.cpu_count * 1e9),
        mem_limit=f"{resources.mem_size}m",
        shm_size=f"{resources.shm_size}m",
    )
    return container.id


def validate_models(db: Session, storage: ModelStorage, model_infos: list) -> list:
    """
    Validates the existence of specified models in the database.

    Args:
        db (Session): The database session.
        storage (ModelStorage): The model storage manager.
        model_infos (list): List of model information to validate.

    Returns:
        list: List of validated model instances.

    Raises:
        HTTPException: If a specified model does not exist.
    """
    model_instances = []
    for model_name in model_infos:
        model = get_single_model(db=db, model_name=model_name, storage=storage)
        assert model is not None, f"Model '{model_name}' does not exist"
        model_instances.append(model)
    return model_instances


def get_available_gpus(db: Session, required_gpus: int) -> list:
    """
    Retrieves available GPUs based on the required amount.

    Args:
        db (Session): The database session.
        required_gpus (int): The number of GPUs required.

    Returns:
        list: List of available GPU devices.

    Raises:
        AssertionError: If not enough GPUs are available.
    """
    if required_gpus > 0:
        device_infos = get_available_devices(db, count=required_gpus)
        if len(device_infos) < required_gpus:
            raise AssertionError(
                f"Not enough GPUs available. Requested: {required_gpus}, Available: {len(device_infos)}"
            )
        return device_infos
    return []


def create_service_entry(
    db: Session,
    service_name: str,
    image_name: str,
    service_timeout: int,
    service_priority: int,
    service_resources: ServiceCreateResources,
    service_environment: dict,
    model_instances: list[Model],
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

    Returns:
        Service: The created service entry.
    """
    service = Service(
        service_name=service_name,
        service_image=image_name,
        inactivity_timeout=service_timeout,
        priority=service_priority,
        container_status=ServiceStatus.STARTING,
        created_at=datetime.now(tz=timezone.utc),
        last_active_time=datetime.now(tz=timezone.utc),
    )
    service.models.extend(model_instances)

    resources = ServiceResources(
        cpu_count=service_resources.cpu_count,
        mem_size=service_resources.mem_size,
        shm_size=service_resources.shm_size,
        environment_variables=service_environment,
    )
    service.resources = resources

    db.add(service)
    db.flush()
    return service


def create_device_allocations(db: Session, service_id: int, device_infos: list):
    """
    Creates device allocation entries for a service.

    Args:
        db (Session): The database session.
        service_id (int): The ID of the service.
        device_infos (list): List of device information to allocate.
    """
    for device in device_infos:
        allocation = DeviceAllocation(
            device_id=device.uuid,
            service_id=service_id,
            allocation_percentage=1.0,
        )
        db.add(allocation)


def create_service(
    db: Session,
    client: DockerClient,
    traefik: TraefikConfigManager,
    storage: ModelStorage,
    service_name: str,
    image_name: str,
    service_network: str,
    service_models_volume: Path,
    service_url_prefix: str,
    service_environment: dict[str, str],
    service_resources: ServiceCreateResources,
    service_timeout: int,
    service_priority: int,
    service_api_keys: list[str],
    model_infos: list[str],
) -> Service:
    """
    Creates a Triton docker container loading the specified models.

    Args:
        db (Session): The database session.
        client (DockerClient): The Docker client.
        traefik (TraefikConfigManager): The Traefik config manager.
        storage (ModelStorage): The model storage manager.
        service_name (str): The name of the service.
        image_name (str): The name of the Docker image to use.
        service_network (str): The name of the Docker network to use.
        service_models_volume (Path): The path to the model repository or a volume name.
        service_url_prefix (str): The URL prefix to use for the service.
        service_environment (dict[str, str]): The environment variables to pass to the container.
        service_resources (ServiceCreateResources): The resources to use for the container.
        service_timeout (int): The timeout for the service.
        service_priority (int): The priority for the service.
        service_api_keys (list[str]): The list of API keys to use for the service.
        model_infos (list): The list of models to load.

    Returns:
        Service: The created service.

    Raises:
        HTTPException: If the service could not be created.
    """
    try:
        # Validate image
        assert image_name, "No image specified"
        image = get_service_image(client, image_name)
        model_instances = validate_models(db, storage, model_infos)
        device_infos = get_available_gpus(db, service_resources.gpus)
        # Create service entry in database
        service = create_service_entry(
            db=db,
            service_name=service_name,
            image_name=image_name,
            service_timeout=service_timeout,
            service_priority=service_priority,
            service_resources=service_resources,
            service_environment=service_environment,
            model_instances=model_instances,
        )
        # Create device allocations
        create_device_allocations(
            db=db,
            service_id=service.service_id,
            device_infos=device_infos,
        )
        # Spawn docker container
        container_id = spawn_service_container(
            client=client,
            image_id=image.id,
            worker_name=service_name,
            worker_network=service_network,
            worker_volume=service_models_volume,
            models=model_instances,
            devices=device_infos,
            resources=service_resources,
            environment=service_environment,
        )

        # Update service with container ID and configure Traefik
        service.container_id = container_id
        traefik.add(
            service_prefix=service_url_prefix,
            service_name=service.service_name,
            api_keys=service_api_keys,
        )
        # commit changes to be persisted
        db.commit()
        db.refresh(service)
        return service

    except AssertionError as e:
        db.rollback()
        raise HTTPException(status_code=409, detail=f"Error creating service: {str(e)}") from e
    except APIError as e:
        db.rollback()
        raise HTTPException(status_code=e.status_code, detail=f"Error creating service: {str(e)}") from e
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Unexpected error creating service: {str(e)}") from e


def delete_service(
    db: Session,
    client: DockerClient,
    traefik: TraefikConfigManager,
    service_id: int,
) -> Service:
    """
    Deletes a triton docker container and the traefik config for the service.
    Marks the service as deleted, deallocates associated devices, and removes the Docker container.

    Args:
        db (Session): The database session.
        client (DockerClient): The docker client.
        traefik (TraefikConfigManager): The traefik config manager.
        service_id (int): The ID of the service.

    Returns:
        Service: The deleted service.

    Raises:
        HTTPException: If the service doesn't exist or if there's an error during deletion.
    """
    try:
        # check if service exists
        if (service := get_service_by_id(db=db, docker_client=client, service_id=service_id)) is None:
            raise HTTPException(status_code=404, detail=f"Service with id {service_id} does not exist")

        current_time = datetime.now(tz=timezone.utc)

        # Handle the Docker container
        if service.container_id:
            try:
                container = client.containers.get(service.container_id)
                container.remove(force=True)  # This stops and removes the container
            except NotFound:
                LOG.warning(f"Container for service {service_id} not found. It may have been already removed.")
            except APIError as e:
                LOG.error(f"Error removing Docker container for service {service_id}: {str(e)}")
                service.container_status = ServiceStatus.ERROR
                raise HTTPException(status_code=500, detail=f"Error removing Docker container: {str(e)}.")

        # Update service status
        service.deleted_at = current_time
        service.container_status = ServiceStatus.DELETED
        service.container_id = None  # Clear the container ID as it's been removed

        # Remove traefik config
        try:
            traefik.delete(service_name=service.service_name)
        except Exception as e:
            LOG.error(f"Error removing Traefik config for service {service_id}: {str(e)}")

        # Commit the changes
        db.commit()
        db.refresh(service)
        return service

    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        LOG.exception(f"Unexpected error while deleting service {service_id}")
        raise HTTPException(status_code=500, detail=f"Unexpected error deleting service: {str(e)}")


def start_service(
    db: Session,
    client: DockerClient,
    service: Service,
) -> None:
    try:
        LOG.debug(f"Starting service {service.service_id}...")
        client.containers.get(service.container_id).start()
        service.container_status = ServiceStatus.ACTIVE
        service.last_active_time = datetime.now(tz=timezone.utc)
        db.commit()
    except NotFound:
        raise HTTPException(status_code=404, detail="Service not found")
    except Exception as e:
        db.rollback()
        LOG.debug(f"Error starting container: {str(e)}")
        raise HTTPException(status_code=503, detail="Error restarting service")


def update_active_time(db: Session, service: Service):
    """Updates the last active time of a service.

    Args:
        db (Session): The database session.
        service (Service): The service to update.
    """
    service.last_active_time = datetime.now(tz=timezone.utc)
    db.commit()


def stop_service(
    db: Session,
    client: DockerClient,
    service_id: int,
) -> None:
    try:
        LOG.debug(f"Stopping service {service_id}...")
        service = get_service_by_id(db=db, docker_client=client, service_id=service_id)
        client.containers.get(service.container_id).stop()
        service.container_status = ServiceStatus.STOPPED
        db.commit()
    except NotFound:
        raise HTTPException(status_code=404, detail="Service not found")
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Error stopping service: {str(e)}")
