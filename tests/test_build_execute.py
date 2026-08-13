from contextlib import suppress

import pytest
from docker.errors import ImageNotFound
from httpx import Client

from triton_serve.builder.execute import build_image
from triton_serve.builder.registry import image_ref
from triton_serve.builder.spec import make_build_spec
from triton_serve.config.schema import AppSettings
from triton_serve.database.model import ImageStatus, ServiceImage, utcnow

GITHUB_API = "https://api.github.com"


def _delete_scratch_package(settings: AppSettings) -> None:
    """Empties the package these tests push to, so a run leaves the registry as it found it.

    Specific to ghcr: the registry API can push and pull but not delete, so cleanup goes through the
    GitHub packages API. Everything goes rather than only the tags of this run, because the package
    holds nothing but what these tests push -- which is also the only thing that makes deleting from
    it safe -- so leftovers from an interrupted run are cleaned up too. The API refuses to delete a
    package's last version and wants the package itself deleted instead.
    """
    package = f"/orgs/{settings.registry_namespace}/packages/container/{settings.registry_image_name}"
    headers = {
        "Authorization": f"Bearer {settings.registry_push_token.get_secret_value()}",
        "Accept": "application/vnd.github+json",
    }
    with Client(base_url=GITHUB_API, headers=headers, timeout=30) as client:
        versions = client.get(f"{package}/versions", params={"per_page": 100})
        if versions.status_code == 404:
            return
        versions.raise_for_status()
        for version in versions.json():
            response = client.delete(f"{package}/versions/{version['id']}")
            if response.status_code == 400:
                client.delete(package).raise_for_status()
                return
            response.raise_for_status()


@pytest.fixture(scope="session")
def scratch_package(test_settings):
    yield
    if test_settings.registry_image_name.endswith("-test"):
        _delete_scratch_package(test_settings)


@pytest.fixture
def push_credentials(test_settings, scratch_package):
    if not test_settings.registry_push_token.get_secret_value():
        pytest.skip("No registry push credentials configured")
    if not test_settings.registry_image_name.endswith("-test"):
        pytest.skip(f"Refusing to build into the shared package '{test_settings.registry_image_name}'")
    yield test_settings


@pytest.fixture
def built_images(test_settings, push_credentials, test_docker):
    """Collects the refs a test builds so they do not pile up on the build host."""
    refs: list[str] = []
    yield refs
    for ref in refs:
        with suppress(ImageNotFound):
            test_docker.images.remove(ref, force=True)


@pytest.fixture
def pending_image(test_db, test_settings, push_credentials, built_images):
    spec = make_build_spec(
        base_image="python:3.12-slim",
        apt_packages=[],
        pip_packages=["six==1.16.0"],
        allowed_index_hosts=test_settings.pip_index_allowed_hosts,
    )
    image = ServiceImage(
        image_hash=spec.image_hash,
        image_ref=image_ref(test_settings, spec.image_hash),
        status=ImageStatus.PENDING,
        managed=True,
        base_image=spec.base_image,
        apt_packages=list(spec.apt_packages),
        pip_packages=list(spec.pip_packages),
        pip_extra_index_urls=[],
        created_at=utcnow(),
    )
    test_db.merge(image)
    test_db.commit()
    built_images.append(image.image_ref)
    yield image
    test_db.query(ServiceImage).filter(ServiceImage.image_hash == image.image_hash).delete()
    test_db.commit()


def test_build_image_marks_the_row_ready(test_db, pending_image, test_docker):
    build_image(pending_image.image_hash)
    test_db.expire_all()
    row = test_db.get(ServiceImage, pending_image.image_hash)
    assert row.status is ImageStatus.READY
    assert row.built_at is not None
    assert test_docker.images.get(row.image_ref) is not None


def test_build_from_a_private_base_image_authenticates_the_pull(
    test_db, test_settings, pending_image, test_docker, built_images
):
    """Every real service builds FROM a private base, and `build` takes no per-call credentials."""
    build_image(pending_image.image_hash)
    test_db.expire_all()
    private_base = test_db.get(ServiceImage, pending_image.image_hash).image_ref
    test_docker.images.remove(private_base, force=True)

    spec = make_build_spec(
        base_image=private_base,
        apt_packages=[],
        pip_packages=["six==1.16.0"],
        allowed_index_hosts=test_settings.pip_index_allowed_hosts,
    )
    derived = ServiceImage(
        image_hash=spec.image_hash,
        image_ref=image_ref(test_settings, spec.image_hash),
        status=ImageStatus.PENDING,
        managed=True,
        base_image=spec.base_image,
        apt_packages=list(spec.apt_packages),
        pip_packages=list(spec.pip_packages),
        pip_extra_index_urls=[],
        created_at=utcnow(),
    )
    test_db.merge(derived)
    test_db.commit()
    built_images.append(derived.image_ref)
    try:
        build_image(derived.image_hash)
        test_db.expire_all()
        assert test_db.get(ServiceImage, derived.image_hash).status is ImageStatus.READY
    finally:
        test_db.query(ServiceImage).filter(ServiceImage.image_hash == derived.image_hash).delete()
        test_db.commit()


def test_build_image_marks_a_broken_spec_failed(test_db, pending_image):
    row = test_db.get(ServiceImage, pending_image.image_hash)
    row.pip_packages = ["this-package-does-not-exist-93f2a1=="]
    row.status = ImageStatus.PENDING
    test_db.commit()
    # the terminal branch is what is under test, so the retry budget starts spent: celery's own
    # retry policy is the framework's concern, not this code's
    build_image.push_request(retries=build_image.max_retries)
    try:
        build_image(row.image_hash)
    finally:
        build_image.pop_request()
    test_db.expire_all()
    row = test_db.get(ServiceImage, row.image_hash)
    assert row.status is ImageStatus.FAILED
    assert row.build_log
