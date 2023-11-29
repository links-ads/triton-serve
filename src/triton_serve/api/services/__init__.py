from docker import DockerClient
from fastapi import APIRouter, Depends

from triton_serve.api.schema import ServiceCreateSchema
from triton_serve.api.services import domain
from triton_serve.config import (
    AppSettings,
    TraefikConfigManager,
    get_settings,
    get_traefik,
)
from triton_serve.extensions import docker_client

router = APIRouter()


@router.post("/services", status_code=201, tags=["services"])
async def post_service(
    service: ServiceCreateSchema,
    docker_client: DockerClient = Depends(docker_client),
    settings: AppSettings = Depends(get_settings),
    traefik: TraefikConfigManager = Depends(get_traefik),
):
    """
    Creates a new service with the specified models.

    **Arguments:**
    - `service` (`ServiceCreateSchema`): The service to be created.

    **Returns:**
    - `Service` (`ServiceCreateSchema`): The created service.
    """
    domain.create_service(
        client=docker_client,
        traefik=traefik,
        service_name=service.name,
        image_name=settings.service_image,
        base_command=settings.service_command,
        service_network=settings.service_network,
        service_url_prefix=settings.service_prefix,
        service_models_volume=settings.service_volume,
        models=service.models,
        repository_path=settings.repository_path,
    )


@router.delete("/services/{name}", status_code=204, tags=["services"])
async def delete_service(
    name: str,
    docker_client: DockerClient = Depends(docker_client),
    settings: AppSettings = Depends(get_settings),
    traefik: TraefikConfigManager = Depends(get_traefik),
):
    """
    Deletes a service with the specified name.

    **Arguments:**
    - `name` (`str`): The name of the service to be deleted.

    **Returns:**
    - `None`
    """
    domain.delete_service(
        client=docker_client,
        traefik=traefik,
        service_name=name,
    )
