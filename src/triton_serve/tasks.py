import logging
from datetime import datetime

from celery.signals import worker_process_init, worker_process_shutdown
from httpx import Client
from sqlalchemy import text
from sqlalchemy.orm import joinedload

from triton_serve.api.services.execute import execute
from triton_serve.api.services.observe import observe
from triton_serve.api.services.reconcile import decide
from triton_serve.builder.execute import build_image  # noqa: F401  (registers the task on the app)
from triton_serve.config import get_settings
from triton_serve.config.celery import client as worker_client
from triton_serve.database import database_manager
from triton_serve.database.model import DesiredState, RuntimeStatus, Service, timezone_aware_now
from triton_serve.extensions import get_reconciler_docker_client
from triton_serve.queue import app

LOG = logging.getLogger(__name__)

# single-flight guard: only one reconcile pass may touch Docker at a time. an overrunning tick
# (slow daemon, image pull) must never run concurrently with the next and both decide RECREATE.
_RECONCILE_LOCK_KEY = 0x7213_10CE

settings = get_settings()


@worker_process_init.connect
def init_worker_database(**_):
    """The webserver's lifespan never runs here, so the worker owns its own engine.

    Per forked child, not at import: the prefork parent's connections would be inherited by every
    child and used concurrently on the same sockets.
    """
    database_manager.init(settings.database_url)


@worker_process_shutdown.connect
def close_worker_database(**_):
    database_manager.close()


@app.on_after_configure.connect  # type: ignore
def setup_periodic_tasks(sender, **_):
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
    """How many seconds before retrying"""
    return min(base * (2 ** max(attempts - 1, 0)), cap)


@app.task
def update_service_status() -> None:
    """Reconcile every non-retired service: observe -> decide -> execute. Reconciler owns Docker."""
    client = get_reconciler_docker_client()
    # single-flight across the whole pass: hold the advisory lock on a dedicated connection so it
    # survives the per-service commits below -- an ORM session releases its connection on each commit,
    # which would strand the lock on a pooled connection and unlock a different one. autocommit keeps
    # that connection from sitting idle-in-transaction, pinning a snapshot for the whole pass
    with database_manager.connect(isolation_level="AUTOCOMMIT") as lock_conn:
        if not lock_conn.execute(text("SELECT pg_try_advisory_lock(:k)"), {"k": _RECONCILE_LOCK_KEY}).scalar():
            LOG.info("another reconcile pass holds the advisory lock; skipping this tick")
            return
        try:
            with database_manager.session() as db:
                # joinedload: the tick reads service.image.status for every service, and an
                # N+1 per tick is exactly what this loop must not do
                services = (
                    db.query(Service)
                    .options(joinedload(Service.image))
                    .filter(Service.runtime_status != RuntimeStatus.RETIRED)
                    .all()
                )
                for service in services:
                    try:
                        now = timezone_aware_now()
                        # honor backoff: skip a service mid-retry still cooling down between attempts.
                        # RECOVERING is the status the executor persists after any spent attempt (crash
                        # recreate or image pull), so image-pull failures draw on the same crash budget
                        if (
                            service.runtime_status == RuntimeStatus.RECOVERING
                            and service.restart_attempts > 0
                            and service.last_attempt_at is not None
                            and (now - service.last_attempt_at).total_seconds()
                            < _backoff_seconds(
                                service.restart_attempts,
                                base=settings.service_restart_backoff_base,
                                cap=settings.service_restart_cooldown,
                            )
                        ):
                            continue

                        target = (
                            _replica_target(service, now) if service.desired_state == DesiredState.AVAILABLE else 0
                        )
                        image_status = service.image.status if service.image is not None else None
                        observed = observe(client, service, settings.service_boot_grace, image_status)
                        decision = decide(
                            desired=service.desired_state,
                            observed=observed,
                            replica_target=target,
                            attempts=service.restart_attempts,
                            max_attempts=settings.service_max_restart_attempts,
                        )
                        LOG.debug(
                            "reconcile %s: desired=%s target=%d observed=%s attempts=%d -> %s => %s",
                            service.service_name,
                            service.desired_state.value,
                            target,
                            observed.value,
                            service.restart_attempts,
                            decision.action.value,
                            decision.status.value,
                        )
                        execute(db=db, client=client, service=service, decision=decision, settings=settings)
                    except Exception as e:
                        LOG.error("Reconcile failed for service %s: %s", service.service_id, e)
                        db.rollback()
        finally:
            lock_conn.execute(text("SELECT pg_advisory_unlock(:k)"), {"k": _RECONCILE_LOCK_KEY})


@app.task
def purge_queue_messages(client: Client | None = None) -> None:
    """Purges queue messages older than the configured window.

    The beat schedule fires this with no arguments, so `client` must stay optional; it exists for
    tests to inject their own. The worker has no client in a test environment, where the backend
    it would call is the process under test.
    """
    client = client or worker_client
    if client is None:
        LOG.warning("No backend client configured, skipping the queue purge")
        return
    try:
        response = client.delete("queue/messages")
        LOG.debug("Purge of queue messages complete: %s", response.text)
    except Exception as e:
        LOG.error("Error purging queue messages: %s", e)
