import logging
import secrets
from datetime import timedelta

from fastapi import HTTPException
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from triton_serve.database.model import APIKey, KeyType, Service, utcnow

LOG = logging.getLogger("uvicorn")


def get_key(db: Session, key_id: int) -> APIKey:
    """
    Retrieve an API key by its ID.

    Args:
        db (Session): SQLAlchemy session
        key_id (int): API key ID

    Returns:
        APIKey: API key object if found, else None
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
        List[APIKey]: List of API keys
    """
    try:
        query = db.query(APIKey)
        if key_type:
            query = query.filter_by(key_type=key_type)
        if project:
            query = query.filter_by(project=project)
        if service:
            query = query.join(APIKey.services).filter_by(service_name=service)
        return query.all()
    except SQLAlchemyError as e:
        raise HTTPException(status_code=500, detail=e._message())


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
    expires_at = utcnow() + timedelta(days=expiration_days)
    new_key = APIKey(
        value=key,
        key_type=key_type,
        project=project,
        notes=notes,
        expires_at=expires_at,
    )
    try:
        if services:
            new_key.services = services

        db.add(new_key)
        db.commit()
        return new_key
    except SQLAlchemyError as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=e._message())


def revoke_key(db: Session, key: str):
    """
    Revoke an API key by its value.

    Args:
        db (Session): SQLAlchemy session
        key (str): API key value
    """
    try:
        api_key = db.query(APIKey).filter_by(value=key).first()
        if api_key:
            db.delete(api_key)
            db.commit()
        else:
            raise HTTPException(status_code=404, detail="Key not found")
    except SQLAlchemyError as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=e._message())


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
    try:
        api_key = db.query(APIKey).filter_by(value=key).first()
        if not api_key:
            raise HTTPException(status_code=404, detail="Key not found")

        if project is not None:
            api_key.project = project
        if notes is not None:
            api_key.notes = notes

        db.commit()
        db.refresh(api_key)
        return api_key
    except SQLAlchemyError as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))


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
    try:
        service_ids = [s.service_id for s in key.services]
        if service.service_id not in service_ids:
            key.services.append(service)
            db.commit()
        else:
            raise HTTPException(status_code=400, detail="Service already added to key")
        db.refresh(key)
        return key
    except SQLAlchemyError as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=e._message())


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
    try:
        updated_services = [s for s in key.services if s.service_id != service.service_id]
        key.services = updated_services
        db.commit()
        db.refresh(key)
        return key
    except SQLAlchemyError as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=e._message())
