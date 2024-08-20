from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from triton_serve.api.auth.domain import (
    add_service_to_key,
    generate_key,
    get_key,
    list_keys,
    remove_service_from_key,
    revoke_key,
    update_key,
)
from triton_serve.api.dto import (
    APIKeyCreateBody,
    APIKeyUpdateBody,
    KeyType,
    ServiceKeyCreateBody,
)
from triton_serve.config import get_traefik
from triton_serve.config.traefik import TraefikConfigManager
from triton_serve.database.schema import APIKeySchema
from triton_serve.extensions import get_db
from triton_serve.security import require_admin
from triton_serve.api.services.domain import get_service_by_id

router = APIRouter()


@router.get(
    "/keys",
    status_code=200,
    tags=["keys"],
    response_model=list[APIKeySchema],
)
def list_api_keys(
    key_type: KeyType = None,
    project: str = None,
    service: str = None,
    db: Session = Depends(get_db),
    _: Any = Depends(require_admin),
):
    """
    Retrieves a list of API keys, allowing filtering by `key_type`, `project`, and `service`.

    **Arguments:**
    - `key_type` (`Optional[KeyType]`, optional): Type of the key. Defaults to `None`.
    - `project` (`Optional[str]`, optional): Project name. Defaults to `None`.
    - `service` (`Optional[str]`, optional): Service name. Defaults to `None`.

    **Returns:**
    - `List[APIKeySchema]`: A list of API keys.
    """
    return list_keys(
        db=db,
        key_type=key_type,
        project=project,
        service=service,
    )


@router.post(
    "/keys",
    status_code=201,
    tags=["keys"],
    response_model=APIKeySchema,
)
def create_api_key(
    key_data: APIKeyCreateBody,
    db: Session = Depends(get_db),
    _: Any = Depends(require_admin),
):
    """
    Creates a new API key for users or admins. For services, use the `/keys/{service_id}` endpoint.

    **Arguments:**
    - `key_data` (`APIKeyCreateBody`): Data to create a new API key.

    **Returns:**
    - `APIKeySchema`: The newly created API key.
    """
    if key_data.key_type == KeyType.SERVICE:
        raise HTTPException(status_code=400, detail="Use /keys/{service_id} for service key creation")

    new_key = generate_key(
        db=db,
        project=key_data.project,
        key_type=key_data.key_type,
        notes=key_data.notes,
        expiration_days=key_data.expiration_days,
    )
    return new_key


@router.put(
    "/keys/{key}",
    status_code=200,
    tags=["keys"],
    response_model=APIKeySchema,
)
def update_api_key(
    key: str,
    update: APIKeyUpdateBody,
    db: Session = Depends(get_db),
    _: Any = Depends(require_admin),
):
    """
    Updates an existing API key.

    **Arguments:**
    - `key` (`str`, optional): The key to update. Defaults to `None`.
    - `project` (`str`, optional): The project name. Defaults to `None`.
    - `notes` (`str`, optional): Additional notes. Defaults to `None`.

    **Returns:**
    - `APIKeySchema`: The updated API key.
    """
    return update_key(
        db=db,
        key=key,
        project=update.project,
        notes=update.notes,
    )


@router.delete(
    "/keys/{key}",
    status_code=204,
    tags=["keys"],
)
def revoke_api_key(
    key: str,
    db: Session = Depends(get_db),
    _: Any = Depends(require_admin),
):
    """
    Revokes an existing API key.

    **Arguments:**
    - `key` (`str`): The key to revoke.

    **Returns:**
    - `None`
    """
    revoke_key(db=db, key=key)


@router.post("/keys/{service_id}", status_code=201, tags=["keys"], response_model=APIKeySchema)
def create_service_key(
    service_id: int,
    key_data: ServiceKeyCreateBody,
    db: Session = Depends(get_db),
    traefik: TraefikConfigManager = Depends(get_traefik),
    _: Any = Depends(require_admin),
):
    """
    Creates a new API key for a specific service.
    """
    # Check if the service exists
    if not (service := get_service_by_id(db=db, service_id=service_id)):
        raise HTTPException(status_code=404, detail="Service not found")

    new_key = generate_key(
        db=db,
        project=key_data.project,
        key_type=KeyType.SERVICE,
        notes=key_data.notes,
        expiration_days=key_data.expiration_days,
        services=[service],
    )

    # Update Traefik configuration with the new key
    traefik.add_service_key(service.service_name, new_key.value)
    return new_key


@router.post("/keys/{key_id}/services/{service_id}", status_code=200, tags=["keys"], response_model=APIKeySchema)
def add_service_key(
    key_id: int,
    service_id: int,
    db: Session = Depends(get_db),
    traefik: TraefikConfigManager = Depends(get_traefik),
    _: Any = Depends(require_admin),
):
    """
    Adds a service to an existing API key.
    """
    key = get_key(db, key_id)
    if not key:
        raise HTTPException(status_code=404, detail="Key not found")

    if key.key_type != KeyType.SERVICE:
        raise HTTPException(status_code=400, detail="This operation is only valid for service keys")

    if not (service := get_service_by_id(db=db, service_id=service_id)):
        raise HTTPException(status_code=404, detail="Service not found")

    updated_key = add_service_to_key(db=db, key=key, service=service)
    traefik.add_service_key(service.service_name, updated_key.value)

    return updated_key


@router.delete("/keys/{key_id}/services/{service_id}", status_code=204, tags=["keys"])
def remove_service_key(
    key_id: int,
    service_id: int,
    db: Session = Depends(get_db),
    traefik: TraefikConfigManager = Depends(get_traefik),
    _: Any = Depends(require_admin),
):
    """
    Removes a service from an existing API key.
    """
    key = get_key(db=db, key_id=key_id)
    if not key:
        raise HTTPException(status_code=404, detail="Key not found")

    if key.key_type != KeyType.SERVICE:
        raise HTTPException(status_code=400, detail="This operation is only valid for service keys")

    if not (service := get_service_by_id(db=db, service_id=service_id)):
        raise HTTPException(status_code=404, detail="Service not found")

    if service_id not in (s.service_id for s in key.services):
        raise HTTPException(status_code=400, detail="This key is not associated with the specified service")

    remove_service_from_key(db=db, key=key, service=service)
    traefik.remove_service_key(service.service_name, key.value)
