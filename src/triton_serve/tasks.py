import logging
from datetime import datetime, timedelta

from celery.signals import worker_process_init, worker_process_shutdown
from sqlalchemy.orm import joinedload

from triton_serve.api.services.execute import execute
from triton_serve.api.services.observe import observe
from triton_serve.api.services.reconcile import decide
from triton_serve.builder.execute import (  # noqa: F401  (registers the tasks on the app)
    build_image,
    reap_stale_builds,
)
from triton_serve.config import get_settings, get_storage
from triton_serve.database import database_manager
from triton_serve.database.model import DesiredState, Model, ModelVersion, RuntimeStatus, Service, timezone_aware_now
from triton_serve.extensions import get_reconciler_docker_client
from triton_serve.queue import app
from triton_serve.storage.base import StoredVersion

LOG = logging.getLogger(__name__)

# single-flight guard: only one reconcile pass may touch Docker at a time. an overrunning tick
# (slow daemon, image pull) must never run concurrently with the next and both decide RECREATE.
_RECONCILE_LOCK_KEY = 0x7213_10CE
_SWEEP_LOCK_KEY = 0x7213_5EEB

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

    # the default queue, not BUILDER_QUEUE: there it would queue behind the long build it watches
    sender.add_periodic_task(
        settings.sentinel_poll_interval,
        reap_stale_builds.s(),  # type: ignore
        name="Reap stale image builds",
    )

    sender.add_periodic_task(
        settings.repository_sweep_interval,
        sweep_orphaned_files.s(),  # type: ignore
        name="Sweep orphaned model files",
    )


def _replica_target(service: Service, now: datetime) -> int:
    """1 if within the inactivity window (recent traffic / wake), else 0. AVAILABLE only."""
    idle_for = (now - service.last_active_time).total_seconds()
    return 1 if idle_for < service.inactivity_timeout else 0


def _backoff_seconds(attempts: int, base: int, cap: int) -> int:
    """How many seconds before retrying"""
    return min(base * (2 ** max(attempts - 1, 0)), cap)


def _orphans(stored: list[StoredVersion], known: set[tuple[str, int]], cutoff: datetime) -> list[StoredVersion]:
    """Picks the stored versions no database row accounts for and that are older than the cutoff.

    Returns nothing when the database reports no rows at all: an empty repository is a legitimate
    state, but an empty database beside a full repository is an outage or a truncated read, and
    acting on it would delete every model on the platform.
    """
    if stored and not known:
        LOG.error("sweep found %d stored versions and no database rows; deleting nothing", len(stored))
        return []
    return [v for v in stored if (v.model_name, v.version_id) not in known and v.modified_at < cutoff]


@app.task
def update_service_status() -> None:
    """Reconcile every non-retired service: observe -> decide -> execute. Reconciler owns Docker."""
    client = get_reconciler_docker_client()
    # single-flight across the whole pass: an overrunning tick must never run concurrently with the
    # next and both decide RECREATE
    with database_manager.advisory_lock(_RECONCILE_LOCK_KEY) as acquired:
        if not acquired:
            LOG.info("another reconcile pass holds the advisory lock; skipping this tick")
            return
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

                    target = _replica_target(service, now) if service.desired_state == DesiredState.AVAILABLE else 0
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


@app.task
def sweep_orphaned_files() -> None:
    """Deletes stored model versions that no `model_versions` row accounts for.

    The backstop for what #136's in-process undo cannot reach: a crash between the files landing and
    the rows committing leaves files with no owner, and against a blob container those are invisible
    rather than something anyone trips over.
    """
    settings = get_settings()
    storage = get_storage()
    cutoff = timezone_aware_now() - timedelta(seconds=settings.orphan_min_age)
    with database_manager.advisory_lock(_SWEEP_LOCK_KEY) as acquired:
        if not acquired:
            LOG.info("another sweep holds the advisory lock; skipping this tick")
            return
        stored = storage.list_versions()
        with database_manager.session() as db:
            known = {
                (name, version_id)
                for name, version_id in db.query(Model.model_name, ModelVersion.version_id).join(
                    ModelVersion, Model.model_id == ModelVersion.model_id
                )
            }
        for version in _orphans(stored, known, cutoff):
            try:
                storage.delete(version, version)
                LOG.warning("swept orphaned version %s:%d", version.model_name, version.version_id)
            except Exception as e:
                # a backstop must never be the thing that kills a beat worker
                LOG.error("could not sweep %s:%d: %s", version.model_name, version.version_id, e)
