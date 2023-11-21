from pathlib import Path

import yaml
from docker import DockerClient
from docker.errors import APIError
from fastapi import HTTPException


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
        raise HTTPException(status_code=500, detail=f"Error creating container: {e}")


def update_traefik_config(service_prefix: str, service_name: str, config_path: Path):
    prefix_name = f"{service_prefix}/{service_name}"
    path_prefix = f"PathPrefix(`{prefix_name}`)"
    service_url = f"http://{service_name}:8000"

    yaml_file_name = config_path / f"{service_name}.yaml"
    raw_data = {
        "http": {
            "services": {service_name: {"loadBalancer": {"servers": [{"url": service_url}]}}},
            "middlewares": {
                f"{service_name}_sablier": {
                    "plugin": {
                        "sablier": {
                            "group": "serve-workers",
                            "names": service_name,
                            "sablierUrl": "http://sablier:10000",
                            "sessionDuration": "1m",
                            "blocking": {
                                "timeout": "30s",
                            },
                        }
                    }
                },
                f"{service_name}_stripprefix": {"stripPrefix": {"prefixes": [prefix_name]}},
            },
            "routers": {
                service_name: {
                    "rule": path_prefix,
                    "entryPoints": ["http"],
                    "middlewares": [f"{service_name}_sablier@file", f"{service_name}_stripprefix@file"],
                    "service": service_name,
                }
            },
        }
    }

    with open(yaml_file_name, "w") as file:
        yaml.dump(raw_data, file)


def create_service(
    client: DockerClient,
    service_name: str,
    image_name: str,
    base_command: str,
    service_network: str,
    service_url_prefix: str,
    service_models_volume: Path,
    models: list[str],
    configs_path: Path,
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
        str: The id of the created container.

    Raises:
        HTTPException: If the container could not be created.
    """
    spawn_worker_container(
        client=client,
        image_name=image_name,
        worker_name=service_name,
        worker_command=base_command,
        worker_network=service_network,
        models=models,
        worker_volume=service_models_volume,
    )
    update_traefik_config(service_prefix=service_url_prefix, service_name=service_name, config_path=configs_path)
