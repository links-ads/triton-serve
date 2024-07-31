import logging
from datetime import datetime, timezone

from celery import Celery
from dateutil.parser import isoparse
from httpx import Client

from triton_serve.config import get_settings
from triton_serve.database.model import ServiceStatus

settings = get_settings()
client = Client(
    base_url=f"http://{settings.backend_host}:{settings.backend_port}",
    headers={"X-API-Key": settings.sentinel_api_key},
)

app = Celery("tasks")
app.config_from_object("triton_serve.config.celery")

LOG = logging.getLogger(__name__)


def check_inactivity(service: dict) -> bool:
    """
    Checks if the service needs to be stopped

    Args:
        service (dict): The service to check

    Returns:
        bool: True if the service needs to be stopped, False otherwise
    """
    up_time = (datetime.now(tz=timezone.utc) - isoparse(service["last_active_time"])).total_seconds()
    if up_time > service["inactivity_timeout"]:
        return True
    return False


@app.task
def check_and_update_container_task() -> None:
    """
    Checks the status of the container.

    Args:
        backend_host (str): The backend host
        backend_port (str): The backend port
        api_key (str): The API key

    Returns:
        None
    """
    try:
        response = client.get("/services")
        json_response = response.json()
        for service in json_response:
            if service["container_status"] == ServiceStatus.ACTIVE.value and check_inactivity(service):
                client.post(f'/services/{service["service_id"]}/stop')
    except Exception as e:
        print(e)
        LOG.error("Error checking container status: %s", e)
