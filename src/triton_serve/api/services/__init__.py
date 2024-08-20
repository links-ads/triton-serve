from typing import Any

from docker import DockerClient
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from triton_serve.api.dto import ServiceCreateBody
from triton_serve.api.services import domain
from triton_serve.config import (
    AppSettings,
    TraefikConfigManager,
    get_settings,
    get_storage,
    get_traefik,
)
from triton_serve.database.model import ServiceStatus
from triton_serve.database.schema import ServiceSchema
from triton_serve.extensions import docker_client, get_db
from triton_serve.security import require_admin, require_elevated, require_service
from triton_serve.storage import ModelStorage

router = APIRouter()


@router.get(
    "/services",
    status_code=200,
    tags=["services"],
    response_model=list[ServiceSchema],
)
def get_services(
    names: list[str] = Query(None),
    statuses: list[ServiceStatus] = Query(None),
    db: Session = Depends(get_db),
    docker: DockerClient = Depends(docker_client),
    _: Any = Depends(require_elevated),
):
    """
    Retrieves a list of services.

    **Arguments:**
    - `names` (`Optional[list[str]]`, optional): Names of the services to be retrieved. Defaults to `None`.
    - `statuses` (`Optional[list[ServiceStatus]]`, optional): Status of the service. Defaults to `None`.

    **Returns:**
    - `List[Service]`: A list of services.
    """
    services = domain.list_services(
        db=db,
        docker_client=docker,
        names=names,
        statuses=statuses,
    )
    return services


@router.get(
    "/services/{service_id}",
    status_code=200,
    tags=["services"],
    response_model=ServiceSchema,
)
def get_service(
    service_id: int,
    db: Session = Depends(get_db),
    docker: DockerClient = Depends(docker_client),
    _: Any = Depends(require_elevated),
):
    """
    Retrieves a specific service by name, if present.

    **Arguments:**
    - `name` (`str`): The name of the service.

    **Returns:**
    - `Service`: The requested service.
    """
    service = domain.get_service_by_id(
        db=db,
        docker_client=docker,
        service_id=service_id,
    )
    if service is None:
        raise HTTPException(status_code=404, detail=f"Service with ID={service_id} does not exist")
    return service


@router.post(
    "/services",
    status_code=201,
    tags=["services"],
    response_model=ServiceSchema,
)
def create_service(
    service_params: ServiceCreateBody,
    settings: AppSettings = Depends(get_settings),
    docker: DockerClient = Depends(docker_client),
    traefik: TraefikConfigManager = Depends(get_traefik),
    storage: ModelStorage = Depends(get_storage),
    db: Session = Depends(get_db),
    _: Any = Depends(require_admin),
):
    """
    Creates a new service with the specified models.

    **Arguments:**
    - `name` (`string`): The name of the service to be created.
    - `models` (`list[Model]`): The models to be served by the service.
    - `docker_image` (`Optional[str]`): The docker image to be used for the service. Defaults to `tritonserver:23.07-py3`.
    - `environment` (`Optional[dict]`): Environment variables to be passed to the service. Defaults to `{}`.
    - `resources` (`Optional[ServiceResources]`): Resources to be allocated to the service.
    - `timeout` (`Optional[int]`): Timeout for the service. Defaults to `3600`.
    - `priority` (`Optional[int]`): Priority of the service. Defaults to `1`.

    **Returns:**
    - `Service` (`ServiceCreateSchema`): Information about the created service.
    """
    docker_image = service_params.docker_image or settings.service_default_image
    return domain.create_service(
        client=docker,
        traefik=traefik,
        storage=storage,
        service_name=service_params.name,
        image_name=docker_image,
        service_network=settings.service_network,
        service_url_prefix=settings.service_prefix,
        service_models_volume=settings.service_volume,
        service_environment=service_params.environment,
        service_resources=service_params.resources,
        service_timeout=service_params.timeout,
        service_priority=service_params.priority,
        service_api_keys=settings.api_keys,
        model_infos=service_params.models,
        db=db,
    )


@router.delete(
    "/services/{service_id}",
    status_code=202,
    tags=["services"],
    response_model=ServiceSchema,
)
def delete_service(
    service_id: int,
    docker: DockerClient = Depends(docker_client),
    traefik: TraefikConfigManager = Depends(get_traefik),
    db: Session = Depends(get_db),
    _: Any = Depends(require_admin),
):
    """
    Deletes a service with the specified name.

    **Arguments:**
    - `service_id` (`int`): The id of the service to be deleted.

    **Returns:**
    - `None`
    """
    return domain.delete_service(
        db=db,
        client=docker,
        traefik=traefik,
        service_id=service_id,
    )


@router.post(
    "/services/{service_id}/stop",
    status_code=204,
    tags=["services"],
)
def stop_service(
    service_id: int,
    docker: DockerClient = Depends(docker_client),
    db: Session = Depends(get_db),
):
    """
    Stops the container of a service with the specified id.

    **Arguments:**
    - `service_id` (`int`): The id of the service to be stopped.

    **Returns:**
    - `None`
    """
    domain.stop_service(
        db=db,
        client=docker,
        service_id=service_id,
    )


@router.get(
    "/status/{service_name}",
    status_code=200,
    tags=["status"],
)
def check_service_status(
    service_name: str,
    db: Session = Depends(get_db),
    docker: DockerClient = Depends(docker_client),
    _: Any = Depends(require_service),
):
    """
    Checks the status of a service, turning it on if it is stopped.

    **Returns:**
    - `200` if the service is running,
    - `202` if the service is starting,
    - `503` if the service is cannot be started.
    """
    print(f"Checking status of service: {service_name}")
    service = domain.get_service_by_name(
        db=db,
        service_name=service_name,
        docker_client=docker,
    )
    match service.container_status:
        case ServiceStatus.ACTIVE:
            print(f"Service '{service_name}' is active")
            domain.update_active_time(db=db, service=service)
            return 200
        case ServiceStatus.STARTING:
            print(f"Service '{service_name}' is starting")
            domain.update_active_time(db=db, service=service)
            return 202
        case ServiceStatus.STOPPED:
            print(f"Service '{service_name}' is stopped")
            domain.start_service(
                db=db,
                client=docker,
                service=service,
            )
            return 202
        case ServiceStatus.ERROR:
            raise HTTPException(status_code=503, detail=f"Service '{service_name}' is in error state")
        case _:
            raise HTTPException(status_code=503, detail=f"Service '{service_name}' is unavailable")
