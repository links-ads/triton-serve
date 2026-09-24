import pytest
import sqlalchemy as sa
from sqlalchemy.orm import Session

from triton_serve import factory
from triton_serve.config.traefik import TraefikConfigManager
from triton_serve.database.model import (
    APIKey,
    Base,
    Service,
    key_service_association,
    timezone_aware_now,
)


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


@pytest.fixture
def sync_settings(tmp_path, monkeypatch, test_settings):
    """Points the sync at a scratch config directory instead of the real one."""
    monkeypatch.setattr(factory, "get_traefik", lambda: TraefikConfigManager(tmp_path))
    return test_settings.model_copy(update={"configs_path": tmp_path})


def test_the_sync_writes_a_config_for_a_live_service(db_session, sync_settings, service):
    factory.sync_traefik_configs(db_session, sync_settings)

    assert (sync_settings.configs_path / "demo.yaml").exists()


def test_the_sync_removes_a_config_with_no_live_service(db_session, sync_settings, service):
    """Five of these accumulated in production: the sync only ever added, never removed."""
    orphan = sync_settings.configs_path / "gone.yaml"
    orphan.write_text("http: {}\n")

    factory.sync_traefik_configs(db_session, sync_settings)

    assert not orphan.exists()
    assert (sync_settings.configs_path / "demo.yaml").exists()


def test_the_sync_leaves_files_it_does_not_own_alone(db_session, sync_settings, service):
    """Traefik reads .yaml; a .bak someone left behind is not ours to delete."""
    backup = sync_settings.configs_path / "gone.yaml.bak"
    backup.write_text("http: {}\n")

    factory.sync_traefik_configs(db_session, sync_settings)

    assert backup.exists()


def test_no_live_services_deletes_nothing(db_session, sync_settings):
    """An empty result beside a full directory is an outage, not a mandate to drop every route."""
    orphan = sync_settings.configs_path / "survivor.yaml"
    orphan.write_text("http: {}\n")

    factory.sync_traefik_configs(db_session, sync_settings)

    assert orphan.exists()
