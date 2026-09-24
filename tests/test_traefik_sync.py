import pytest
import sqlalchemy as sa
from sqlalchemy.orm import Session

from triton_serve.config.traefik import TraefikConfigManager
from triton_serve.database.model import Base, Service, timezone_aware_now
from triton_serve.factory import sync_traefik_configs


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
def traefik(tmp_path):
    return TraefikConfigManager(tmp_path)


def test_the_sync_writes_a_config_for_a_live_service(db_session, traefik, test_settings, service):
    sync_traefik_configs(db_session, traefik, test_settings)

    assert (traefik.configs_path / "demo.yaml").exists()


def test_the_sync_removes_a_config_with_no_live_service(db_session, traefik, test_settings, service):
    """The sync only ever added, so a service deleted while the API was down kept its route forever."""
    orphan = traefik.configs_path / "gone.yaml"
    orphan.write_text("http: {}\n")

    sync_traefik_configs(db_session, traefik, test_settings)

    assert not orphan.exists()
    assert (traefik.configs_path / "demo.yaml").exists()


def test_the_sync_leaves_files_it_does_not_own_alone(db_session, traefik, test_settings, service):
    """Traefik reads .yaml; a .bak someone left behind is not ours to delete."""
    backup = traefik.configs_path / "gone.yaml.bak"
    backup.write_text("http: {}\n")

    sync_traefik_configs(db_session, traefik, test_settings)

    assert backup.exists()


def test_no_live_services_deletes_nothing(db_session, traefik, test_settings):
    """An empty result beside a full directory is an outage, not a mandate to drop every route."""
    orphan = traefik.configs_path / "survivor.yaml"
    orphan.write_text("http: {}\n")

    sync_traefik_configs(db_session, traefik, test_settings)

    assert orphan.exists()
