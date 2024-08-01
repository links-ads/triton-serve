import logging
from datetime import datetime, timezone

from celery import Celery
from httpx import Client

from triton_serve.config import get_settings
from triton_serve.config.celery import Config
from triton_serve.config.celery import client as worker_client
from triton_serve.database.model import ServiceStatus
from triton_serve.database.schema import ServiceSchema

LOG = logging.getLogger(__name__)

settings = get_settings()
app = Celery("serve-sentinel")
app.config_from_object(Config)


@app.on_after_configure.connect
def setup_periodic_tasks(sender, **_):
    """
    Setup periodic tasks
    """
    sender.add_periodic_task(
        settings.sentinel_poll_interval,
        update_service_status.s(),
        name="Update service status",
    )


@app.task
def update_service_status(client: Client = None) -> None:
    """
    Checks the status of the container.
    """
    client = client or worker_client
    try:
        response = client.get("/services")
        json_response = response.json()
        for service in json_response:
            service = ServiceSchema(**service)
            if service.container_status != ServiceStatus.ACTIVE:
                continue
            up_time = (datetime.now(tz=timezone.utc) - service.last_active_time).total_seconds()
            LOG.debug("Service %s has been inactive for %s seconds", service.service_id, up_time)
            if up_time > service.inactivity_timeout:
                LOG.info("Stopping service %s due to inactivity", service.service_id)
                response = client.post(f"/services/{service.service_id}/stop")
                LOG.debug("Service %s stopped: %s", service.service_id, response.text)
    except Exception as e:
        LOG.error("Error checking container status: %s", e)
