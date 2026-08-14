import logging
import secrets
from datetime import timedelta

from fastapi import HTTPException
from sqlalchemy.orm import Session

from triton_serve.database.model import APIKey, KeyType, Service, timezone_aware_now

LOG = logging.getLogger("uvicorn")


def get_key(db: Session, key_id: int) -> APIKey | None:
    """
    Retrieve an API key by its ID.

    Args:
        db (Session): SQLAlchemy session
        key_id (int): API key ID

    Returns:
        APIKey | None: API key object if found, None otherwise
    """
    return db.query(APIKey).filter(APIKey.key_id == key_id).first()


def list_keys(
    db: Session,
    key_type: KeyType | None = None,
    project: str | None = None,
    service: str | None = None,
):
    """
    Retrieve a list of API keys, allowing filtering by `key_type`, `project`, and `service`.

    Args:
        db (Session): SQLAlchemy session
        key_type (KeyType): Type of the key
        project (str): Project name
        service (str): Service name

    Returns:
        list[APIKey]: List of API keys
    """
    query = db.query(APIKey)
    if key_type:
        query = query.filter_by(key_type=key_type)
    if project:
        query = query.filter_by(project=project)
    if service:
        query = query.join(APIKey.services).filter_by(service_name=service)
    return query.all()


def generate_key(
    db: Session,
    key_type: KeyType,
    project: str | None = None,
    notes: str | None = None,
    expiration_days: int = 30,
    services: list[Service] | None = None,
) -> APIKey:
    """
    Generate a new API key, with an optional expiration date and services.

    Args:
        db (Session): SQLAlchemy session
        key_type (KeyType): Type of the key
        project (str): Project name
        notes (str): Additional notes
        expiration_days (int): Number of days until the key expires
        services (list[Service]): List of services

    Returns:
        APIKey: Newly created API key.
    """
    key = secrets.token_urlsafe(32)
    expires_at = timezone_aware_now() + timedelta(days=expiration_days)
    new_key = APIKey(
        value=key,
        key_type=key_type,
        project=project,
        notes=notes,
        expires_at=expires_at,
    )
    if services:
        new_key.services = services

    db.add(new_key)
    db.commit()
    return new_key


def revoke_key(db: Session, key: str):
    """
    Revoke an API key by its value.

    Args:
        db (Session): SQLAlchemy session
        key (str): API key value
    """
    api_key = db.query(APIKey).filter_by(value=key).first()
    if api_key is None:
        raise HTTPException(status_code=404, detail="Key not found")
    db.delete(api_key)
    db.commit()


def update_key(
    db: Session,
    key: str,
    project: str | None,
    notes: str | None,
) -> APIKey:
    """
    Update an existing API key with new project and notes.

    Args:
        db (Session): SQLAlchemy session
        key (str): API key value
        project (str): New project name
        notes (str): New notes

    Returns:
        APIKey: Updated API key.
    """
    LOG.debug(f"Updated info: {project}, {notes}")
    api_key = db.query(APIKey).filter_by(value=key).first()
    if api_key is None:
        raise HTTPException(status_code=404, detail="Key not found")

    if project is not None:
        api_key.project = project
    if notes is not None:
        api_key.notes = notes

    db.commit()
    db.refresh(api_key)
    return api_key


def add_service_to_key(db: Session, key: APIKey, service: Service) -> APIKey:
    """
    Add a service to an existing API key.

    Args:
        db (Session): SQLAlchemy session
        key (APIKey): API key object
        service (Service): Service object

    Returns:
        APIKey: Updated API key.
    """
    service_ids = [s.service_id for s in key.services]
    if service.service_id in service_ids:
        raise HTTPException(status_code=400, detail="Service already added to key")
    key.services.append(service)
    db.commit()
    db.refresh(key)
    return key


def remove_service_from_key(db: Session, key: APIKey, service: Service) -> APIKey:
    """
    Remove a service from an existing API key.

    Args:
        db (Session): SQLAlchemy session
        key (APIKey): API key object
        service (Service): Service object

    Returns:
        APIKey: Updated API key.
    """
    key.services = [s for s in key.services if s.service_id != service.service_id]
    db.commit()
    db.refresh(key)
    return key
