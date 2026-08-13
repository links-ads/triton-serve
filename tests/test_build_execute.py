import pytest

from triton_serve.builder.execute import build_image
from triton_serve.database.model import ImageStatus, ServiceImage, utcnow


@pytest.fixture
def push_credentials(test_settings):
    if not test_settings.registry_push_token.get_secret_value():
        pytest.skip("No registry push credentials configured")
    yield test_settings


@pytest.fixture
def pending_image(test_db, test_settings, push_credentials):
    from triton_serve.builder.registry import image_ref
    from triton_serve.builder.spec import make_build_spec

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
