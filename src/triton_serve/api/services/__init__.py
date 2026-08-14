from typing import Any

from fastapi import APIRouter, Depends, Query, Response
from sqlalchemy.orm import Session

from triton_serve.api.dto import ServiceCreateBody, ServiceUpdateBody
from triton_serve.api.services import domain
from triton_serve.config import (
    AppSettings,
    TraefikConfigManager,
    get_settings,
    get_traefik,
)
from triton_serve.database.model import DesiredState, RuntimeStatus
from triton_serve.database.schema import ServiceSchema
from triton_serve.extensions import get_db
from triton_serve.security import require_admin, require_elevated, require_service

router = APIRouter()

# retry cadence handed to traefik/clients while a service is not yet ready: one reconcile tick,
# so a caller retries roughly once per chance the reconciler has to bring the service up
_RETRY_AFTER = str(get_settings().sentinel_poll_interval)


@router.get(
    "/services",
    status_code=200,
    tags=["services"],
    response_model=list[ServiceSchema],
)
def get_services(
    names: list[str] = Query(None),
    runtime_statuses: list[RuntimeStatus] = Query(None),
    db: Session = Depends(get_db),
    _: Any = Depends(require_elevated),
):
    """
    Retrieves a list of services.

    **Arguments:**
    - `names` (`Optional[list[str]]`, optional): Names of the services to be retrieved. Defaults to `None`.
    - `runtime_statuses` (`Optional[list[RuntimeStatus]]`, optional): Runtime status filter. Defaults to `None`.

    **Returns:**
    - `List[Service]`: A list of services.
    """
    return domain.list_services(db=db, names=names, runtime_statuses=runtime_statuses)


@router.get(
    "/services/{service_id}",
    status_code=200,
    tags=["services"],
    response_model=ServiceSchema,
)
def get_service(
    service_id: int,
    db: Session = Depends(get_db),
    _: Any = Depends(require_elevated),
):
    """
    Retrieves a specific service by id. A deleted service counts as absent.

    **Arguments:**
    - `service_id` (`int`): The id of the service.

    **Returns:**
    - `Service`: The requested service.
    """
    return domain.get_service_or_not_found(db=db, service_id=service_id)


@router.get(
    "/services/{service_id}/config",
    status_code=200,
    tags=["services"],
    response_model=ServiceCreateBody,
)
def get_service_config(
    service_id: int,
    db: Session = Depends(get_db),
    _: Any = Depends(require_elevated),
):
    """
    Returns the creation config of a service as a `ServiceCreateBody`.

    The response can be used as-is (or modified) to recreate the same service via `POST /services`.

    **Arguments:**
    - `service_id` (`int`): The id of the service.

    **Returns:**
    - `ServiceCreateBody`: The service creation config.
    """
    return domain.get_service_config(db=db, service_id=service_id)


@router.post(
    "/services",
    status_code=201,
    tags=["services"],
    response_model=ServiceSchema,
)
def create_service(
    service_params: ServiceCreateBody,
    settings: AppSettings = Depends(get_settings),
    traefik: TraefikConfigManager = Depends(get_traefik),
    db: Session = Depends(get_db),
    _: Any = Depends(require_admin),
):
    """
    Declaratively creates a new service. The reconciler spawns the container out of band.

    Returns `201` before the container exists; the service comes up on the reconciler's next tick.
    A bad image reference surfaces asynchronously as a `FAILED` runtime status, not a create error.

    **Arguments:**
    - `name` (`string`): The name of the service to be created.
    - `models` (`list[Model]`): The models to be served by the service.
    - `docker_image` (`Optional[str]`): The docker image to be used for the service.
    - `environment` (`Optional[dict]`): Environment variables to be passed to the service. Defaults to `{}`.
    - `resources` (`Optional[ServiceResources]`): Resources to be allocated to the service.
    - `timeout` (`Optional[int]`): Timeout for the service. Defaults to `3600`.
    - `priority` (`Optional[int]`): Priority of the service. Defaults to `1`.
    - `healthcheck` (`Optional[ServiceHealthcheck]`): Container healthcheck. Without one, the service
      is reported `READY` once it has been up for `service_boot_grace` seconds, regardless of whether
      it can actually serve.

    **Returns:**
    - `Service` (`ServiceSchema`): Information about the created service.
    """
    docker_image = service_params.docker_image or settings.service_default_image
    return domain.create_service(
        db=db,
        traefik=traefik,
        settings=settings,
        service_name=service_params.name,
        image_name=docker_image,
        service_url_prefix=settings.service_prefix,
        service_environment=service_params.environment,
        service_resources=service_params.resources,
        service_timeout=service_params.timeout,
        service_priority=service_params.priority,
        model_infos=service_params.models,
        service_api_keys=settings.api_keys,
        service_healthcheck=service_params.healthcheck,
    )


@router.delete(
    "/services/{service_id}",
    status_code=204,
    tags=["services"],
)
def delete_service(
    service_id: int,
    traefik: TraefikConfigManager = Depends(get_traefik),
    db: Session = Depends(get_db),
    _: Any = Depends(require_admin),
):
    """
    Soft-deletes a service: removes its Traefik config now and marks it RETIRED; the reconciler
    tears down the container out of band.

    **Arguments:**
    - `service_id` (`int`): The id of the service to be deleted.

    **Returns:**
    - `None`
    """
    domain.delete_service(db=db, traefik=traefik, service_id=service_id)


@router.post("/services/{service_id}/suspend", status_code=204, tags=["operations"])
def suspend_service(service_id: int, db: Session = Depends(get_db), _: Any = Depends(require_admin)):
    """
    Operator intent: keep the service off (no auto-wake). The reconciler stops it.

    **Arguments:**
    - `service_id` (`int`): The id of the service to suspend.

    **Returns:**
    - `None`
    """
    domain.set_desired_state(db=db, service_id=service_id, desired=DesiredState.SUSPENDED)


@router.post("/services/{service_id}/resume", status_code=204, tags=["operations"])
def resume_service(service_id: int, db: Session = Depends(get_db), _: Any = Depends(require_admin)):
    """
    Operator intent: make the service available again; records a wake so it comes up.

    **Arguments:**
    - `service_id` (`int`): The id of the service to resume.

    **Returns:**
    - `None`
    """
    domain.set_desired_state(db=db, service_id=service_id, desired=DesiredState.AVAILABLE, wake=True)


@router.post("/services/{service_id}/retry", status_code=204, tags=["operations"])
def retry_service(service_id: int, db: Session = Depends(get_db), _: Any = Depends(require_admin)):
    """
    Manual escape hatch for a FAILED service: reset the crash budget and wake it.

    **Arguments:**
    - `service_id` (`int`): The id of the service to retry.

    **Returns:**
    - `None`
    """
    domain.reset_and_wake(db=db, service_id=service_id)


@router.get(
    "/status/{service_name}",
    tags=["operations"],
    responses={
        200: {"description": "Service is READY; forward the request."},
        404: {"description": "No such service, or it is RETIRED."},
        503: {"description": "Service is not ready (warming, idle, recovering, suspended, or failed)."},
    },
)
def service_status(
    service_name: str,
    db: Session = Depends(get_db),
    _: Any = Depends(require_service),
) -> Response:
    """traefik forwardAuth hook. Reads the persisted runtime_status ONLY -- never Docker.

    Not-ready never returns 2XX (a 2XX makes forwardAuth forward to a dead backend).
    IDLE additionally records wake intent; the reconciler brings the service up out of band.
    """
    service = domain.get_service_record_by_name(db=db, service_name=service_name)
    if service is None or service.runtime_status == RuntimeStatus.RETIRED:
        return Response(status_code=404)

    match service.runtime_status:
        case RuntimeStatus.READY:
            domain.update_active_time(db=db, service=service)
            return Response(status_code=200)
        case RuntimeStatus.IDLE:
            domain.update_active_time(db=db, service=service)  # wake intent -> replica_target=1 next tick
            return Response(status_code=503, headers={"Retry-After": _RETRY_AFTER})
        case RuntimeStatus.WARMING | RuntimeStatus.RECOVERING:
            # a client still polling through a slow boot keeps the service wanted, so the
            # reconciler does not scale it back to zero mid-wake once inactivity elapses
            domain.update_active_time(db=db, service=service)
            return Response(status_code=503, headers={"Retry-After": _RETRY_AFTER})
        case _:  # SUSPENDED, FAILED
            return Response(status_code=503)


@router.put("/services/{service_id}", status_code=200, tags=["services"], response_model=ServiceSchema)
def update_service(
    service_id: int,
    update_params: ServiceUpdateBody,
    db: Session = Depends(get_db),
    settings: AppSettings = Depends(get_settings),
    _: Any = Depends(require_admin),
):
    """
    Updates service parameters (declarative). Changes take effect on the next (re)create.

    **Arguments:**
    - `service_id` (`int`): The id of the service to update.
    - `update_params` (`ServiceUpdateBody`): Partial update payload (all fields optional).

    **Returns:**
    - `Service`: The updated service.
    """
    return domain.update_service(db=db, service_id=service_id, update_body=update_params, settings=settings)
