import logging
from datetime import datetime, timezone

from celery import Celery
from httpx import Client

from triton_serve.api.services.execute import execute
from triton_serve.api.services.observe import observe
from triton_serve.api.services.reconcile import decide
from triton_serve.config import get_settings
from triton_serve.config.celery import Config
from triton_serve.config.celery import client as worker_client
from triton_serve.database import database_manager
from triton_serve.database.model import DesiredState, RuntimeStatus, Service
from triton_serve.extensions import get_reconciler_docker_client

LOG = logging.getLogger(__name__)

settings = get_settings()
app = Celery("serve-sentinel")
app.config_from_object(Config)


@app.on_after_configure.connect  # type: ignore
def setup_periodic_tasks(sender, **_):
    """
    Setup periodic tasks
    """
    sender.add_periodic_task(
        settings.sentinel_poll_interval,
        update_service_status.s(),  # type: ignore
        name="Update service status",
    )

    sender.add_periodic_task(
        settings.purge_message_schedule,
        purge_queue_messages.s(),  # type: ignore
        name="Purge queue messages",
    )


def _replica_target(service: Service, now: datetime) -> int:
    """1 if within the inactivity window (recent traffic / wake), else 0. AVAILABLE only."""
    idle_for = (now - service.last_active_time).total_seconds()
    return 1 if idle_for < service.inactivity_timeout else 0


def _backoff_seconds(attempts: int, base: int, cap: int) -> int:
    return min(base * (2 ** max(attempts - 1, 0)), cap)


@app.task
def update_service_status() -> None:
    """Reconcile every non-retired service: observe -> decide -> execute. Reconciler owns Docker."""
    client = get_reconciler_docker_client()
    now = datetime.now(tz=timezone.utc)
    with database_manager.session() as db:
        services = db.query(Service).filter(Service.runtime_status != RuntimeStatus.RETIRED).all()
        for service in services:
            try:
                # reset the crash budget once the cooldown has elapsed since the last attempt
                if (
                    service.last_attempt_at is not None
                    and (now - service.last_attempt_at).total_seconds() > settings.service_restart_cooldown
                ):
                    service.restart_attempts = 0

                # honor backoff: skip a service mid-retry that is still cooling down between attempts.
                # RECOVERING is the status the executor persists after any spent attempt (crash recreate
                # or image pull), so image-pull failures draw on the same budget as crash loops
                if (
                    service.runtime_status == RuntimeStatus.RECOVERING
                    and service.restart_attempts > 0
                    and service.last_attempt_at is not None
                    and (now - service.last_attempt_at).total_seconds()
                    < _backoff_seconds(service.restart_attempts, base=10, cap=settings.service_restart_cooldown)
                ):
                    continue

                target = _replica_target(service, now) if service.desired_state == DesiredState.AVAILABLE else 0
                observed = observe(client, service, settings.service_boot_grace)
                decision = decide(
                    desired=service.desired_state,
                    observed=observed,
                    replica_target=target,
                    attempts=service.restart_attempts,
                    max_attempts=settings.service_max_restart_attempts,
                )
                execute(db=db, client=client, service=service, decision=decision, settings=settings)
            except Exception as e:
                LOG.error("Reconcile failed for service %s: %s", service.service_id, e)
                db.rollback()


@app.task
def purge_queue_messages(client: Client | None) -> None:
    """
    Purge queue messages
    """
    client = client or worker_client
    assert client is not None
    try:
        response = client.delete("queue/messages")
        LOG.debug("Purge of queue messages complete: %s", response.text)
    except Exception as e:
        LOG.error("Error purging queue messages: %s", e)
