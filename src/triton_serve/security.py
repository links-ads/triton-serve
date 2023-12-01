from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader

from triton_serve.config.instance import settings

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=True)


async def api_key_auth(api_key: str = Security(api_key_header)):
    if api_key != settings.app_secret:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API Key")
