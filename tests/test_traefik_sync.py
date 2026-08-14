from datetime import timedelta

import pytest
import sqlalchemy as sa
import yaml
from sqlalchemy.orm import Session

from triton_serve.config.traefik import TraefikConfigManager
from triton_serve.database.model import (
    APIKey,
    Base,
    KeyType,
    Service,
    key_service_association,
    timezone_aware_now,
)

DEFAULT_KEYS = ["master-key"]


@pytest.fixture
def db_session():
    engine = sa.create_engine("sqlite://")
    # create only the tables involved here, the rest rely on postgres-only types
    Base.metadata.create_all(engine, tables=[APIKey.__table__, Service.__table__, key_service_association])
    with Session(engine) as session:
        yield session


@pytest.fixture
def traefik(tmp_path):
    return TraefikConfigManager(tmp_path)


@pytest.fixture
def service(db_session):
    svc = Service(
        service_name="demo",
        service_image="img",
        last_active_time=timezone_aware_now(),
        priority=0,
    )
    db_session.add(svc)
    db_session.commit()
    return svc


def _read_keys(config_path) -> list[str]:
    with open(config_path) as file:
        config = yaml.safe_load(file)
    name = config_path.stem
    middlewares = config["http"]["middlewares"]
    return middlewares[f"{name}-auth"]["plugin"]["traefik-api-key-middleware"]["keys"]


def _associate_key(db: Session, service: Service, value: str) -> APIKey:
    key = APIKey(
        key_type=KeyType.SERVICE,
        value=value,
        project="test",
        expires_at=timezone_aware_now() + timedelta(days=30),
    )
    key.services.append(service)
    db.add(key)
    db.commit()
    return key


def test_assigned_service_key_persists_across_config_rebuilds(db_session, traefik, service):
    from triton_serve.api.services.domain import rebuild_service_config

    # service creation writes only the default/master keys (mirrors create_service)
    traefik.add(service_prefix="", service_name=service.service_name, api_keys=list(DEFAULT_KEYS))

    # operator associates an existing service key -> database is updated
    _associate_key(db_session, service, "svc-key")

    # the assignment rebuilds the config from database truth ...
    rebuild_service_config(db_session, traefik, service, service_prefix="", default_keys=DEFAULT_KEYS)
    # ... and every later rebuild (create / refresh / startup sync) must preserve it
    rebuild_service_config(db_session, traefik, service, service_prefix="", default_keys=DEFAULT_KEYS)

    keys = _read_keys(traefik.configs_path / f"{service.service_name}.yaml")
    assert "svc-key" in keys
    assert "master-key" in keys
