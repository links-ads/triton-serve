import pytest

from triton_serve.builder.resolve import resolve_service_image, service_build_spec
from triton_serve.config import get_settings
from triton_serve.database.model import ImageStatus, Model, Service, ServiceResources, utcnow

BASE = "ghcr.io/links-ads/serve-triton:23.07-py3"


@pytest.fixture
def settings():
    return get_settings()


@pytest.fixture
def rollback(test_db):
    """These tests write services and images; none of it should survive the test."""
    yield
    test_db.rollback()


def _service(db, name, dependencies, system_dependencies=()):
    service = Service(service_name=name, service_image=BASE, last_active_time=utcnow(), priority=1)
    service.models.append(
        Model(model_name=f"{name}-model", dependencies=dependencies, system_dependencies=list(system_dependencies))
    )
    service.resources = ServiceResources(cpu_count=1, shm_size=64, mem_size=256)
    db.add(service)
    db.flush()
    return service


def test_empty_dependencies_resolve_to_a_ready_row_without_a_build(test_db, settings, rollback):
    service = _service(test_db, "resolve-empty", [])
    assert resolve_service_image(test_db, service, settings) is None
    assert service.image is not None
    assert service.image.status is ImageStatus.READY
    assert service.image.image_ref == BASE
    assert not service.image.managed


def test_dependencies_resolve_to_a_pending_row_needing_a_build(test_db, settings, rollback):
    service = _service(test_db, "resolve-deps", ["numpy==1.26.0"])
    assert resolve_service_image(test_db, service, settings) == service.image_hash
    assert service.image.status is ImageStatus.PENDING
    assert service.image.managed
    assert service.image.image_ref.startswith(
        f"{settings.registry_url}/{settings.registry_namespace}/{settings.registry_image_name}:"
    )


def test_two_services_with_the_same_dependencies_share_one_row(test_db, settings, rollback):
    first = _service(test_db, "resolve-share-a", ["numpy==1.26.0"])
    second = _service(test_db, "resolve-share-b", ["numpy==1.26.0"])
    assert resolve_service_image(test_db, first, settings) is not None
    assert resolve_service_image(test_db, second, settings) is None
    assert first.image_hash == second.image_hash


def test_build_spec_unions_dependencies_across_models(test_db, rollback):
    service = _service(test_db, "resolve-union", ["numpy==1.26.0"], ["libgl1"])
    service.models.append(
        Model(model_name="resolve-union-2", dependencies=["pillow==10.0.0"], system_dependencies=["libsndfile1"])
    )
    test_db.flush()
    spec = service_build_spec(service)
    assert spec.pip_packages == ("numpy==1.26.0", "pillow==10.0.0")
    assert spec.apt_packages == ("libgl1", "libsndfile1")


def test_system_dependencies_change_the_hash(test_db, rollback):
    plain = _service(test_db, "resolve-nosys", ["numpy==1.26.0"])
    with_apt = _service(test_db, "resolve-sys", ["numpy==1.26.0"], ["libgl1"])
    assert service_build_spec(plain).image_hash != service_build_spec(with_apt).image_hash


def test_invalid_stored_dependency_raises(test_db, settings, rollback):
    service = _service(test_db, "resolve-bad", ["--extra-index-url http://attacker.example"])
    with pytest.raises(ValueError, match="requirement"):
        resolve_service_image(test_db, service, settings)
