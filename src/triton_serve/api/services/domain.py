from datetime import datetime, timezone
from pathlib import Path

from docker import DockerClient
from docker.errors import APIError, ImageNotFound, NotFound
from docker.types import DeviceRequest
from fastapi import HTTPException
from sqlalchemy.orm import Session

from triton_serve.api.models.domain import get_model
from triton_serve.config.traefik import TraefikConfigManager
from triton_serve.database.model import Device, Service, ServiceStatus, device_mapping
from triton_serve.database.schema import ServiceCreateSchema
from triton_serve.storage import ModelStorage


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
        statement = statement.filter(Service.name.in_(names))
    if statuses:
        statement = statement.filter(Service.status.in_(statuses))
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


def spawn_worker_container(
    client: DockerClient,
    image_name: str,
    worker_name: str,
    worker_command: str,
    worker_network: str,
    worker_volume: str,
    models: list,
    devices: list = None,
    environment: dict[str, str] = None,
):
    """Spawns a triton worker container.

    Args:

        client (DockerClient): The docker client.
        image_name (str): The name of the docker image to use.
        worker_name (str): The name of the worker container.
        worker_command (str): The command to run the docker image.
        worker_network (str): The name of the docker network to use.
        worker_volume (str): The path to the model repository, or a volume name.
        models (list[str]): The list of models to load.
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
    models_string = " ".join([f"--load-model={model.name}" for model in models])
    command = f"{worker_command} {models_string}"
    labels = {"sablier.enable": "true", "sablier.group": "serve-workers"}
    volumes = {str(worker_volume): {"bind": "/models", "mode": "ro"}}

    environment = environment or {}
    gpus, runtime = None, None
    if devices:
        runtime = "nvidia"
        gpus = [
            DeviceRequest(device_ids=[str(gpu.uuid)], capabilities=[["gpu", "nvidia", "compute"]]) for gpu in devices
        ]

    container = client.containers.run(
        detach=True,
        remove=False,
        image=image_name,
        name=worker_name,
        command=command,
        network=worker_network,
        volumes=volumes,
        environment=environment,
        labels=labels,
        restart_policy={"Name": "unless-stopped"},
        runtime=runtime,
        device_requests=gpus,
    )
    return container.id


def get_free_devices(db: Session, count: int) -> list[Device]:
    """Returns a list of available devices, depending on availability.
    A device is available if it is not assigned to any service in the `device_mapping` table.

    Args:
        db (Session): The database session.
        count (int): The number of GPUs to return.

    Returns:
        list[Device]: The list of available GPUs.
    """
    statement = db.query(Device).outerjoin(device_mapping).filter(device_mapping.c.device_id.is_(None))
    devices = statement.limit(count).all()
    return devices


def persist_service(
    db: Session,
    service_name: str,
    service_image: str,
    models: list,
    devices: list,
    created_at: str,
) -> Service:
    """
    Save the service on the database

    Args:
        service_name (str): the name of the service to save
        models list(str): the list of models to load in the service
        devices list(str): the list of devices to use in the service
        created_at (int): the timestamp of the creation of the service
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
    )
    db_service = Service(**service.model_dump())
    db_service.models.extend(models)
    db_service.devices.extend(devices)
    db.add(db_service)
    return db_service


def create_service(
    db: Session,
    client: DockerClient,
    traefik: TraefikConfigManager,
    storage: ModelStorage,
    service_name: str,
    image_name: str,
    base_command: str,
    service_network: str,
    service_models_volume: Path,
    service_url_prefix: str,
    service_environment: dict[str, str],
    model_infos: list,
    gpu_count: int,
) -> Service:
    """Creates a triton docker container loading the models specified in the models list.

    Args:
        db (Session): The database session.
        client (DockerClient): The docker client.
        traefik (TraefikConfigManager): The traefik config manager.
        service_name (str): The name of the service.
        image_name (str): The name of the docker image to use.
        base_command (str): The base command to run the docker image.
        service_network (str): The name of the docker network to use.
        service_models_volume (Path): The path to the model repository, or a volume name.
        service_url_prefix (str): The url prefix to use for the service.
        service_environment (dict[str, str]): The environment variables to pass to the container (if any).
        model_infos (list[str]): The list of models to load.
        gpu_count (int): how many gpus to use, defaults to 0.

    Returns:
        `service` (`ServiceCreateSchema`): The created service.

    Raises:
        HTTPException: If the container could not be created.
    """
    # assert that the docker image exists
    try:
        client.images.get(image_name)
    except ImageNotFound as e:
        raise HTTPException(status_code=412, detail=f"Docker image {image_name} does not exist") from e
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
        if gpu_count > 0:
            device_infos = get_free_devices(db, count=gpu_count)
            assert device_infos, "No GPU available"
            assert len(device_infos) == gpu_count, f"Not enough GPUs available: {len(device_infos)}"
        timestamp = datetime.now(tz=timezone.utc)
        # persist the service on the database (do not comit yet to avoid inconsistencies)
        service = persist_service(
            db=db,
            service_name=service_name,
            service_image=image_name,
            models=model_instances,
            devices=device_infos,
            created_at=timestamp,
        )
        # spawn the worker container, and save the container id on the database
        container_id = spawn_worker_container(
            client=client,
            image_name=image_name,
            worker_name=service_name,
            worker_command=base_command,
            worker_network=service_network,
            models=model_infos,
            devices=device_infos,
            worker_volume=service_models_volume,
            environment=service_environment,
        )
        # update the service, and the traefik config, then commit
        service.container_id = container_id
        traefik.add(service_prefix=service_url_prefix, service_name=service_name)
        db.commit()
        db.refresh(service)
        return service
    except AssertionError as e:
        raise HTTPException(status_code=409, detail=f"Error creating service: {e}") from e
    except APIError as e:
        raise HTTPException(status_code=e.status_code, detail=f"Error creating service: {e}") from e


def delete_service(
    db: Session,
    client: DockerClient,
    traefik: TraefikConfigManager,
    service_id: int,
    delete_container: bool = False,
) -> Service:
    """Deletes a triton docker container and the traefik config for the service.

    Args:
        db (Session): The database session.
        client (DockerClient): The docker client.
        traefik (TraefikConfigManager): The traefik config manager.
        service_id (int): The ID of the service.
        delete_container (bool, optional): Whether to delete the container or not. Defaults to False.

    Returns:
        `None`

    Raises:
        HTTPException: If the container could not be deleted.
    """
    # check if service exists
    if (service := get_service(db=db, docker_client=client, service_id=service_id)) is None:
        raise HTTPException(status_code=404, detail=f"Service with id {service_id} does not exist")
    # check if the container exists, if not make sure to mark the service as deleted
    try:
        if service.container_id is not None:
            if container := client.containers.get(service.container_id):
                container.stop()
                if delete_container:
                    container.remove()
    except APIError as e:
        service.deleted_at = datetime.now(tz=timezone.utc)
        db.commit()  # commit the deletion of the service, regardless of the container
        raise HTTPException(status_code=e.status_code, detail=f"Error deleting service: {e}") from e

    service.deleted_at = datetime.now(tz=timezone.utc)
    traefik.delete(service_name=service.service_name)
    db.commit()
    db.refresh(service)
    return service
