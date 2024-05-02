from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader

from triton_serve.config import get_settings

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=True)


async def api_key_auth(api_key: str = Security(api_key_header)):
    settings = get_settings()
    if api_key not in settings.api_keys:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API Key")
