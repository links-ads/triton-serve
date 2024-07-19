import logging
from datetime import datetime, timezone
from itertools import chain
from pathlib import Path

from docker import DockerClient
from docker.errors import APIError, ImageNotFound, NotFound
from docker.types import DeviceRequest
from fastapi import HTTPException
from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session

from triton_serve.api.dto import ServiceResources
from triton_serve.api.models.domain import get_model
from triton_serve.config.traefik import TraefikConfigManager
from triton_serve.database.model import (
    Device,
    DeviceAllocation,
    Model,
    Service,
    ServiceStatus,
)
from triton_serve.database.schema import ServiceCreateSchema
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


def get_service(db: Session, docker_client: DockerClient, service_id: int):
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
        .where(DeviceAllocation.deallocated_at is None)
        .group_by(DeviceAllocation.device_id)
        .subquery()
    )

    # Main query to select available devices
    query = (
        select(Device)
        .outerjoin(alloc_subquery, Device.uuid == alloc_subquery.c.device_id)
        .where(
            or_(
                alloc_subquery.c.total_allocation is None,  # Devices with no allocations
                (
                    100 - alloc_subquery.c.total_allocation
                    >= required_percentage  # Devices with enough free allocation
                ),
            )
        )
        .order_by(func.coalesce(alloc_subquery.c.total_allocation, 0))  # Order by least allocated first
        .limit(count)
    )

    return db.scalars(query).all()


def get_service_image(docker_client: DockerClient, image_name: str):
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
            return image.id
        except ImageNotFound:
            image = docker_client.images.pull(image_name)
            return image.id
    except APIError as e:
        raise HTTPException(status_code=412, detail=f"Cannot retrieve image: {e.explanation}") from e


def check_service_status(db: Session, docker_client: DockerClient, service: Service):
    """Checks the status of a service, querying the docker daemon, and updating the database.

    Args:
        db (Session): The database session.
        docker_client (DockerClient): The docker client.
        service (Service): The service to check.
    """
    if service is None:
        return None
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
    except NotFound:
        if service.deleted_at is None:
            service.deleted_at = datetime.now(tz=timezone.utc)
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
    resources: ServiceResources,
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
        resources (ServiceResources): The resources to use for the container.
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
    dependencies_list = list(chain.from_iterable([model.dependencies for model in models]))
    environment["WORKER_REQUIREMENTS"] = " ".join(dependencies_list) if dependencies_list else ""
    print(f"Spawning container {worker_name} with requirements {environment['WORKER_REQUIREMENTS']}")

    # prepare the list of models to load
    triton_args = " ".join([f"--load-model={model.model_name}" for model in models])

    # prepare the labels and volumes for the container
    labels = {"sablier.enable": "true", "sablier.group": "serve-workers"}
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
        labels=labels,
        restart_policy={"Name": "unless-stopped"},
        runtime=runtime,
        device_requests=gpus,
        nano_cpus=int(resources.cpu_count * 1e9),
        mem_limit=f"{resources.mem_size}m",
        shm_size=f"{resources.shm_size}m",
    )
    return container.id


def persist_service(
    db: Session,
    service_name: str,
    service_image: str,
    models: list,
    created_at: str,
    cpu_count: int,
    mem_size: int,
    shm_size: int,
) -> Service:
    """
    Save the service on the database

    Args:
        service_name (str): the name of the service to save
        service_image (str): the image of the service
        models list(str): the list of models to load in the service
        created_at (int): the timestamp of the creation of the service
        cpu_count (int): the number of cpus to use
        mem_size (int): the memory size to use in MB
        shm_size (int): the shared memory size to use in MB
        db (Session): the database session object

    Returns:
        Service: the created service

    Raises:
        ValueError: if no gpu is available

    """

    service = ServiceCreateSchema(
        service_name=service_name,
        service_image=service_image,
        created_at=created_at,
        cpu_count=cpu_count,
        mem_size=mem_size,
        shm_size=shm_size,
    )
    db_service = Service(**service.model_dump())
    db_service.models.extend(models)
    db.add(db_service)
    db.flush()  # flush to get the service_id
    return db_service


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
    service_resources: ServiceResources,
    service_api_keys: list[str],
    model_infos: list,
) -> Service:
    """Creates a triton docker container loading the models specified in the models list.

    Args:
        db (Session): The database session.
        client (DockerClient): The docker client.
        traefik (TraefikConfigManager): The traefik config manager.
        service_name (str): The name of the service.
        image_name (str): The name of the docker image to use.
        service_network (str): The name of the docker network to use.
        service_models_volume (Path): The path to the model repository, or a volume name.
        service_url_prefix (str): The url prefix to use for the service.
        service_environment (dict[str, str]): The environment variables to pass to the container (if any).
        service_resources (`ServiceResources`): The resources to use for the container.
        service_api_keys (list[str]): The list of api keys to use for the service.
        model_infos (list[str]): The list of models to load.

    Returns:
        `service` (`ServiceCreateSchema`): The created service.

    Raises:
        HTTPException: If the container could not be created.
    """
    # assert that the docker image exists, or it can be pulled
    assert image_name, "No image specified"
    image_id = get_service_image(client, image_name)

    # check if the models specified exist
    model_instances = []
    for model_info in model_infos:
        model = get_model(
            db=db,
            model_name=model_info.name,
            model_version=model_info.version,
            storage=storage,
        )
        if model is None:
            raise HTTPException(status_code=409, detail=f"Model <{model_info}> does not exist")
        model_instances.append(model)

    try:
        # retrieve the required amount of available gpus, if requested
        device_infos = []
        if service_resources.gpus > 0:
            device_infos = get_available_devices(db, count=service_resources.gpus)
            if len(device_infos) < service_resources.gpus:
                raise AssertionError(
                    f"Not enough GPUs available. Requested: {service_resources.gpus}, Available: {len(device_infos)}"
                )

        curr_timestamp = datetime.now(tz=timezone.utc)
        # persist the service on the database (do not comit yet to avoid inconsistencies)
        service = persist_service(
            db=db,
            service_name=service_name,
            service_image=image_name,  # for readability
            models=model_instances,
            created_at=curr_timestamp,
            cpu_count=service_resources.cpu_count,
            mem_size=service_resources.mem_size,
            shm_size=service_resources.shm_size,
        )

        # create allocation entries
        for device in device_infos:
            allocation = DeviceAllocation(
                device_id=device.uuid,
                service_id=service.service_id,
                allocation_percentage=1.0,
                allocated_at=curr_timestamp,
            )
            db.add(allocation)

        # spawn the worker container, and save the container id on the database
        container_id = spawn_service_container(
            client=client,
            image_id=image_id,
            worker_name=service_name,
            worker_network=service_network,
            models=model_instances,
            devices=device_infos,
            resources=service_resources,
            worker_volume=service_models_volume,
            environment=service_environment,
        )

        # update the service, and the traefik config, then commit
        service.container_id = container_id
        traefik.add(
            service_prefix=service_url_prefix,
            service_name=service_name,
            api_keys=service_api_keys,
        )
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
        if (service := get_service(db=db, docker_client=client, service_id=service_id)) is None:
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
                # We'll continue with service deletion but include this info in the response
                service.container_status = ServiceStatus.ERROR
                raise HTTPException(
                    status_code=500,
                    detail=f"Error removing Docker container: {str(e)}."
                    "Service marked as deleted but container may still exist.",
                )

        # Update service status
        service.deleted_at = current_time
        service.container_status = ServiceStatus.DELETED
        service.container_id = None  # Clear the container ID as it's been removed

        # Mark device allocations as deallocated
        for allocation in service.devices:
            if allocation.deallocated_at is None:
                allocation.deallocated_at = current_time

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
