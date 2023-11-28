from pathlib import Path

from docker import DockerClient
from docker.errors import APIError
from fastapi import HTTPException

from triton_serve.config.traefik import TraefikConfigManager


def spawn_worker_container(
    client: DockerClient,
    image_name: str,
    worker_name: str,
    worker_command: str,
    worker_network: str,
    worker_volume: str,
    models: list[str],
    environment: dict[str, str] = None,
):
    """Spawns a triton worker container.

    Args:

        client (DockerClient): The docker client.
        image_name (str): The name of the docker image to use.
        worker_name (str): The name of the worker container.
        worker_command (str): The command to run the docker image.
        worker_network (str): The name of the docker network to use.
        models (list[str]): The list of models to load.
        model_repository (str): The path to the model repository, or a volume name.
        environment (dict[str, str], optional): The environment variables to pass to the container. Defaults to None.

    Returns:
        str: The id of the created container.

    Raises:
        HTTPException: If the container could not be created.
    """
    # check if container with the same name already exists
    if worker_name in [container.name for container in client.containers.list()]:
        raise HTTPException(status_code=409, detail=f"Container with name {worker_name} already exists")
    models_string = " ".join([f"--load-model={model}" for model in models])
    command = f"{worker_command} {models_string}"
    labels = {"sablier.enable": "true", "sablier.group": "serve-workers"}
    volumes = {worker_volume: {"bind": "/models", "mode": "ro"}}
    environment = environment or {}

    try:
        container = client.containers.run(
            detach=True,
            image=image_name,
            name=worker_name,
            command=command,
            network=worker_network,
            volumes=volumes,
            environment=environment,
            labels=labels,
            restart_policy={"Name": "unless-stopped"},
        )
        return container.id
    except APIError as e:
        raise HTTPException(status_code=e.status_code, detail=f"Error creating container: {e}")


def create_service(
    client: DockerClient,
    traefik: TraefikConfigManager,
    service_name: str,
    image_name: str,
    base_command: str,
    service_network: str,
    service_url_prefix: str,
    service_models_volume: Path,
    models: list[str],
    repository_path: Path,
):
    """Creates a triton docker container loading the models specified in the models list.

    Args:

        client (DockerClient): The docker client.
        service_name (str): The name of the service.
        image_name (str): The name of the docker image to use.
        base_command (str): The base command to run the docker image.
        service_network (str): The name of the docker network to use.
        service_url_prefix (str): The url prefix to use for the service.
        service_models_volume (Path): The path to the model repository, or a volume name.
        models (list[str]): The list of models to load.
        configs_path (Path): The path to the traefik configs.

    Returns:
        `service` (`ServiceCreateSchema`): The created service.

    Raises:
        HTTPException: If the container could not be created.
    """
    # check if the models specified exist
    for model in models:
        if not (repository_path / model).exists():
            raise HTTPException(status_code=409, detail=f"Model {model} does not exist")
    spawn_worker_container(
        client=client,
        image_name=image_name,
        worker_name=service_name,
        worker_command=base_command,
        worker_network=service_network,
        models=models,
        worker_volume=service_models_volume,
    )
    traefik.add(service_prefix=service_url_prefix, service_name=service_name)


def delete_service(
    client: DockerClient,
    traefik: TraefikConfigManager,
    service_name: str,
):
    """Deletes a triton docker container and the traefik config for the service.

    Args:

        client (DockerClient): The docker client.
        traefik (TraefikConfigManager): The traefik config manager.
        service_name (str): The name of the service.
        configs_path (Path): The path to the traefik configs.

    Returns:
        `None`

    Raises:
        HTTPException: If the container could not be deleted.
    """
    # check if service exists
    if service_name not in [container.name for container in client.containers.list()]:
        raise HTTPException(status_code=404, detail=f"Service with name {service_name} does not exist")
    try:
        container = client.containers.get(service_name)
        container.stop()
        container.remove()
    except APIError as e:
        raise HTTPException(status_code=e.status_code, detail=f"Error deleting service: {e}")
    traefik.delete(service_name=service_name)
