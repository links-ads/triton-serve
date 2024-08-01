from triton_serve.config import get_settings

settings = get_settings()

if settings.environment != "test":
    # only import the client if we are not in a test environment
    from httpx import Client

    client = Client(
        base_url=f"http://{settings.backend_host}:{settings.backend_port}",
        headers={"X-API-Key": settings.api_keys[0]},
    )
else:
    client = None


class Config:
    timezone = "UTC"
    broker_url = settings.celery_broker_url
    # retry to connect to the broker on startup.
    # Not really needed since depends_on is specified in the docker-compose file.
    # It silences a warning.
    broker_connection_retry_on_startup = True
