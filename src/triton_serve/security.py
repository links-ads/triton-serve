from fastapi import Depends, HTTPException, Security
from fastapi.security import APIKeyHeader
from sqlalchemy.orm import Session

from triton_serve.database.model import APIKey, KeyType, utcnow
from triton_serve.extensions import get_db

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=True)


def retrieve_key(db: Session, key: str) -> APIKey:
    api_key = db.query(APIKey).filter_by(value=key).first()
    if api_key and api_key.expires_at > utcnow():
        return api_key
    return None


async def validate_api_key(
    api_key: str = Security(api_key_header),
    db: Session = Depends(get_db),
):
    key = retrieve_key(db=db, key=api_key)
    if not key:
        raise HTTPException(status_code=401, detail="Invalid API Key")
    return key


def require_admin(key: APIKey = Depends(validate_api_key)):
    if key.key_type != KeyType.ADMIN:
        raise HTTPException(status_code=403, detail="Admin access required")
    return key


def require_elevated(key: APIKey = Depends(validate_api_key)):
    if key.key_type not in [KeyType.ADMIN, KeyType.USER]:
        raise HTTPException(status_code=403, detail="Management access required")
    return key


def require_service(key: APIKey = Depends(validate_api_key)):
    if key.key_type not in [KeyType.ADMIN, KeyType.SERVICE]:
        raise HTTPException(status_code=403, detail="Service key required")
    return key
