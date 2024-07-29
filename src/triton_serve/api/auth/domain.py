import secrets
from datetime import timedelta

from sqlalchemy.orm import Session

from triton_serve.database.model import APIKey, KeyType, Service, utcnow


def get_key(db: Session, key_id: int) -> APIKey:
    return db.query(APIKey).filter(APIKey.id == key_id).first()


def list_keys(
    db: Session,
    key_type: KeyType = None,
    project: str = None,
    service: str = None,
):
    query = db.query(APIKey)
    if key_type:
        query = query.filter_by(key_type=key_type)
    if project:
        query = query.filter_by(project=project)
    if service:
        query = query.join(APIKey.services).filter_by(name=service)
    return query.all()


def generate_key(
    db: Session,
    key_type: KeyType,
    project: str = None,
    notes: str = None,
    expiration_days: int = 30,
    services: list[Service] = None,
) -> APIKey:
    key = secrets.token_urlsafe(32)
    expires_at = utcnow() + timedelta(days=expiration_days)
    new_key = APIKey(key=key, key_type=key_type, project=project, notes=notes, expires_at=expires_at)
    if services:
        new_key.services = services

    db.add(new_key)
    db.commit()
    return new_key


def revoke_key(db: Session, key: str):
    api_key = db.query(APIKey).filter_by(key=key).first()
    if api_key:
        db.delete(api_key)
        db.commit()


def update_key(db: Session, key: str, project: str, notes: str) -> APIKey:
    api_key = db.query(APIKey).filter_by(key=key).first()
    if api_key:
        api_key.project = project
        api_key.notes = notes
        db.commit()
        return api_key
    return None


def add_service_to_key(db: Session, key_id: int, service_id: int) -> APIKey:
    key = get_key(key_id)
    service = db.query(Service).filter(Service.id == service_id).first()

    if key and service:
        service_ids = [s.id for s in key.services]
        if service_id not in service_ids:
            key.services.append(service)
            db.commit()
        db.refresh(key)
    return key


def remove_service_from_key(db: Session, key_id: str, service_id: int) -> APIKey:
    key = get_key(key_id)
    if key:
        updated_services = [s for s in key.services if s.id != service_id]
        key.services = updated_services
    db.commit()
    db.refresh(key)
    return key
