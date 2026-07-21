import logging
from datetime import datetime, timezone

from docker import DockerClient
from docker.errors import NotFound
from sqlalchemy.orm import Session

from triton_serve.api.services.domain import get_container_by_name, recreate_service_container
from triton_serve.api.services.reconcile import Action, Decision
from triton_serve.config import get_traefik
from triton_serve.config.schema import AppSettings
from triton_serve.database.model import Service

LOG = logging.getLogger("uvicorn")


def _recreate(db: Session, client: DockerClient, service: Service, settings: AppSettings) -> None:
    recreate_service_container(db, client, service, settings.service_network, settings.service_volume)


def execute(
    db: Session,
    client: DockerClient,
    service: Service,
    decision: Decision,
    settings: AppSettings,
) -> None:
    """Apply a Decision's Action to Docker, then persist the projected runtime_status.

    The only place reconciliation Actions become Docker side effects.
    """
    action = decision.action
    try:
        match action:
            case Action.NONE:
                pass
            case Action.START:
                try:
                    client.containers.get(service.container_id).start()
                except NotFound:
                    _recreate(db, client, service, settings)  # the outage fix: heal a vanished container
            case Action.RECREATE | Action.PULL:
                # docker run pulls a missing image implicitly; PULL and RECREATE share the path
                _recreate(db, client, service, settings)
            case Action.STOP:
                if (c := get_container_by_name(client, service.service_name)) is not None:
                    c.stop()
            case Action.REMOVE:
                if (c := get_container_by_name(client, service.service_name)) is not None:
                    c.remove(force=True)
                service.container_id = None  # type: ignore
            case Action.FINALIZE:
                get_traefik().delete(service_name=service.service_name)
                # allocation release + deleted_at handled by the existing delete path; see domain.delete_service
            case Action.MARK_FAILED:
                LOG.warning("Service %s exhausted restart budget -> FAILED", service.service_id)
    except Exception:
        LOG.exception("Action %s failed for service %s; leaving status for next tick", action, service.service_id)
        db.rollback()
        return

    if decision.increment_attempt:
        service.restart_attempts += 1
        service.last_attempt_at = datetime.now(timezone.utc)
    service.runtime_status = decision.status
    db.commit()
    db.refresh(service)
