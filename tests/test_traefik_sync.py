import pytest
import sqlalchemy as sa
from sqlalchemy.orm import Session

from triton_serve import factory
from triton_serve.config.traefik import TraefikConfigManager
from triton_serve.database.model import Base, Service, timezone_aware_now


@pytest.fixture
def db_session():
    engine = sa.create_engine("sqlite://")
    # create only the tables involved here, the rest rely on postgres-only types
    Base.metadata.create_all(engine, tables=[Service.__table__])
    with Session(engine) as session:
        yield session


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
    return test_settings


def test_the_sync_writes_a_config_for_a_live_service(db_session, sync_settings, service, tmp_path):
    factory.sync_traefik_configs(db_session, sync_settings)

    assert (tmp_path / "demo.yaml").exists()


def test_the_sync_removes_a_config_with_no_live_service(db_session, sync_settings, service, tmp_path):
    """The sync only ever added, so a service deleted while the API was down kept its route forever."""
    orphan = tmp_path / "gone.yaml"
    orphan.write_text("http: {}\n")

    factory.sync_traefik_configs(db_session, sync_settings)

    assert not orphan.exists()
    assert (tmp_path / "demo.yaml").exists()


def test_the_sync_leaves_files_it_does_not_own_alone(db_session, sync_settings, service, tmp_path):
    """Traefik reads .yaml; a .bak someone left behind is not ours to delete."""
    backup = tmp_path / "gone.yaml.bak"
    backup.write_text("http: {}\n")

    factory.sync_traefik_configs(db_session, sync_settings)

    assert backup.exists()


def test_no_live_services_deletes_nothing(db_session, sync_settings, tmp_path):
    """An empty result beside a full directory is an outage, not a mandate to drop every route."""
    orphan = tmp_path / "survivor.yaml"
    orphan.write_text("http: {}\n")

    factory.sync_traefik_configs(db_session, sync_settings)

    assert orphan.exists()
