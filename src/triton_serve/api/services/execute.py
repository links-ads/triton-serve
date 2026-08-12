import logging
from datetime import datetime, timezone

from docker import DockerClient
from sqlalchemy.orm import Session

from triton_serve.api.services.domain import get_container_by_name, recreate_service_container
from triton_serve.api.services.reconcile import Action, Decision
from triton_serve.config.schema import AppSettings
from triton_serve.database.model import RuntimeStatus, Service

LOG = logging.getLogger(__name__)


def _recreate(db: Session, client: DockerClient, service: Service, settings: AppSettings) -> None:
    recreate_service_container(db, client, service, settings.service_network, settings.service_volume)


def _spend_attempt(service: Service) -> None:
    service.restart_attempts += 1
    service.last_attempt_at = datetime.now(timezone.utc)


def _maybe_reset_budget(service: Service, cooldown: int) -> None:
    """Return the crash budget once a recovered service has stayed READY past the cooldown.

    last_attempt_at marks the last recovery attempt; a READY observation never stamps it, so a
    READY service whose last attempt is older than the cooldown has been stably up and earns its
    budget back. FAILED never reaches READY, so this can never revive a terminally failed service.
    """
    if service.restart_attempts and service.last_attempt_at is not None:
        elapsed = (datetime.now(timezone.utc) - service.last_attempt_at).total_seconds()
        if elapsed > cooldown:
            service.restart_attempts = 0
            service.last_attempt_at = None


def execute(
    db: Session,
    client: DockerClient,
    service: Service,
    decision: Decision,
    settings: AppSettings,
) -> None:
    """Apply a Decision's Action to Docker, then persist the projected runtime_status.

    The only place reconciliation Actions become Docker side effects.

    Args:
        db (Session): The database session.
        client (DockerClient): The docker client.
        service (Service): The service the decision was taken for.
        decision (Decision): The action to apply and the status to project.
        settings (AppSettings): The application settings.
    """
    action = decision.action
    try:
        match action:
            case Action.NONE:
                pass
            case Action.START:
                # resolve by name, not the (possibly stale) stored id; recreate if it vanished
                if (c := get_container_by_name(client, service.service_name)) is not None:
                    c.start()
                else:
                    _recreate(db, client, service, settings)
            case Action.RECREATE | Action.PULL:
                # PULL and RECREATE share the path: recreate resolves the image through
                # get_service_image, which falls back to images.pull when it is not present locally
                _recreate(db, client, service, settings)
            case Action.STOP:
                if (c := get_container_by_name(client, service.service_name)) is not None:
                    c.stop()
            case Action.REMOVE:
                if (c := get_container_by_name(client, service.service_name)) is not None:
                    c.remove(force=True)
                service.container_id = None
            case Action.FINALIZE:
                # terminal no-op: traefik config + allocation are torn down synchronously by
                # domain.delete_service; nothing is left to do out of band
                pass
            case Action.MARK_FAILED:
                # decide() returns MARK_FAILED on every tick a failed service is still wanted
                if service.runtime_status != RuntimeStatus.FAILED:
                    LOG.warning("Service %s exhausted restart budget -> FAILED", service.service_id)
    except Exception:
        LOG.exception("Action %s failed for service %s", action, service.service_id)
        db.rollback()
        # a budgeted action (recreate/pull) that raised must still spend the budget, or a broken
        # image would retry forever; RECOVERING makes the next tick honor the backoff gate
        if decision.increment_attempt:
            _spend_attempt(service)
            service.runtime_status = RuntimeStatus.RECOVERING
            db.commit()
            db.refresh(service)
        return

    if decision.increment_attempt:
        _spend_attempt(service)
    elif decision.status == RuntimeStatus.READY:
        _maybe_reset_budget(service, settings.service_restart_cooldown)
    service.runtime_status = decision.status
    db.commit()
    db.refresh(service)
