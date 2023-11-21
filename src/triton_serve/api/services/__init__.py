from docker import DockerClient
from fastapi import APIRouter, Depends

from triton_serve.api.schema import ServiceCreateSchema
from triton_serve.api.services import domain
from triton_serve.config import AppSettings, get_settings
from triton_serve.extensions import docker_client

router = APIRouter()


@router.post("/services", status_code=201, tags=["services"])
async def post_model(
    service: ServiceCreateSchema,
    docker_client: DockerClient = Depends(docker_client),
    settings: AppSettings = Depends(get_settings),
):
    """
    Creates a new service with the specified models.

    **Arguments:**
    - `service` (`ServiceCreateSchema`): The service to be created.

    **Returns:**
    - `Service`: The created service.
    """
    service = domain.create_service(
        client=docker_client,
        service_name=service.name,
        image_name=settings.service_image,
        base_command=settings.service_command,
        service_network=settings.service_network,
        service_url_prefix=settings.service_prefix,
        service_models_volume=settings.service_volume,
        models=service.models,
        configs_path=settings.configs_path,
    )
    return service
