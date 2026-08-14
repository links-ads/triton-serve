from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

from docker.errors import NotFound

from triton_serve.api.services.observe import observe
from triton_serve.api.services.reconcile import ObservedState
from triton_serve.database.model import ImageStatus


def _svc(name="svc", image="img:1"):
    return SimpleNamespace(service_name=name, service_image=image, image=None)


class FakeContainers:
    def __init__(self, container=None):
        self._c = container

    def get(self, name):
        if self._c is None:
            raise NotFound("absent")
        return self._c


class FakeImages:
    def __init__(self, present):
        self._present = present
        self.requested = None

    def get(self, ref):
        from docker.errors import ImageNotFound

        self.requested = ref
        if not self._present:
            raise ImageNotFound(ref)
        return SimpleNamespace(id=ref)


class FakeClient:
    def __init__(self, container=None, image_present=True):
        self.containers = FakeContainers(container)
        self.images = FakeImages(image_present)


def _container(status, exit_code=0, health=None, started=None):
    started = started or datetime.now(UTC)
    return SimpleNamespace(
        status=status,
        attrs={
            "State": {
                "ExitCode": exit_code,
                "StartedAt": started.isoformat(),
                "Health": {"Status": health} if health else {},
            }
        },
    )


def test_absent_with_image_present_is_absent():
    assert (
        observe(FakeClient(container=None, image_present=True), _svc(), 30, ImageStatus.READY) is ObservedState.ABSENT
    )


def test_absent_with_image_missing_is_image_missing():
    assert (
        observe(FakeClient(container=None, image_present=False), _svc(), 30, ImageStatus.READY)
        is ObservedState.IMAGE_MISSING
    )


def test_running_healthy_is_running():
    c = _container("running", health="healthy")
    assert observe(FakeClient(container=c), _svc(), 30, ImageStatus.READY) is ObservedState.RUNNING


def test_running_health_starting_is_booting():
    c = _container("running", health="starting", started=datetime.now(UTC))
    assert observe(FakeClient(container=c), _svc(), 30, ImageStatus.READY) is ObservedState.BOOTING


def test_running_health_unhealthy_is_crashed():
    # docker only reports unhealthy past the start period and after the configured retries, so
    # there is nothing left to wait for: the boot grace must not delay the verdict
    c = _container("running", health="unhealthy", started=datetime.now(UTC))
    assert observe(FakeClient(container=c), _svc(), 30, ImageStatus.READY) is ObservedState.CRASHED


def test_running_health_stuck_past_grace_is_crashed():
    c = _container("running", health="unhealthy", started=datetime.now(UTC) - timedelta(seconds=120))
    assert observe(FakeClient(container=c), _svc(), 30, ImageStatus.READY) is ObservedState.CRASHED


def test_running_health_starting_ignores_boot_grace():
    # starting cannot hang forever, docker leaves it after the start period; calling it crashed on
    # our own clock would recreate a container docker still considers to be warming up
    c = _container("running", health="starting", started=datetime.now(UTC) - timedelta(seconds=120))
    assert observe(FakeClient(container=c), _svc(), 30, ImageStatus.READY) is ObservedState.BOOTING


def test_running_no_health_within_grace_is_booting():
    c = _container("running", started=datetime.now(UTC))
    assert observe(FakeClient(container=c), _svc(), 30, ImageStatus.READY) is ObservedState.BOOTING


def test_running_no_health_past_grace_is_running():
    c = _container("running", started=datetime.now(UTC) - timedelta(seconds=120))
    assert observe(FakeClient(container=c), _svc(), 30, ImageStatus.READY) is ObservedState.RUNNING


def test_exited_zero_is_exited_ok():
    assert (
        observe(FakeClient(container=_container("exited", 0)), _svc(), 30, ImageStatus.READY)
        is ObservedState.EXITED_OK
    )


def test_exited_nonzero_is_crashed():
    assert (
        observe(FakeClient(container=_container("exited", 137)), _svc(), 30, ImageStatus.READY)
        is ObservedState.CRASHED
    )


def test_pending_image_short_circuits_before_docker():
    assert observe(None, _svc(), 30, ImageStatus.PENDING) is ObservedState.IMAGE_PENDING


def test_building_image_short_circuits_before_docker():
    assert observe(None, _svc(), 30, ImageStatus.BUILDING) is ObservedState.IMAGE_PENDING


def test_failed_image_short_circuits_before_docker():
    assert observe(None, _svc(), 30, ImageStatus.FAILED) is ObservedState.IMAGE_FAILED


def test_ready_image_falls_through_to_docker():
    c = _container("running", health="healthy")
    assert observe(FakeClient(container=c), _svc(), 30, ImageStatus.READY) is ObservedState.RUNNING


def test_image_presence_uses_the_resolved_ref():
    service = SimpleNamespace(
        service_name="svc",
        service_image="base:1",
        image=SimpleNamespace(image_ref="ghcr.io/links-ads/serve-runtime:abc123456789"),
    )
    client = FakeClient(container=None, image_present=True)
    assert observe(client, service, 30, ImageStatus.READY) is ObservedState.ABSENT
    assert client.images.requested == "ghcr.io/links-ads/serve-runtime:abc123456789"
