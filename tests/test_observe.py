from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from docker.errors import NotFound

from triton_serve.api.services.observe import observe
from triton_serve.api.services.reconcile import ObservedFact


def _svc(name="svc", image="img:1"):
    return SimpleNamespace(service_name=name, service_image=image)


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

    def get(self, ref):
        from docker.errors import ImageNotFound

        if not self._present:
            raise ImageNotFound(ref)
        return SimpleNamespace(id=ref)


class FakeClient:
    def __init__(self, container=None, image_present=True):
        self.containers = FakeContainers(container)
        self.images = FakeImages(image_present)


def _container(status, exit_code=0, health=None, started=None):
    started = started or datetime.now(timezone.utc)
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
    assert observe(FakeClient(container=None, image_present=True), _svc(), 30) is ObservedFact.ABSENT


def test_absent_with_image_missing_is_image_missing():
    assert observe(FakeClient(container=None, image_present=False), _svc(), 30) is ObservedFact.IMAGE_MISSING


def test_running_healthy_is_running():
    c = _container("running", health="healthy")
    assert observe(FakeClient(container=c), _svc(), 30) is ObservedFact.RUNNING


def test_running_health_starting_is_booting():
    c = _container("running", health="starting", started=datetime.now(timezone.utc))
    assert observe(FakeClient(container=c), _svc(), 30) is ObservedFact.BOOTING


def test_running_no_health_within_grace_is_booting():
    c = _container("running", started=datetime.now(timezone.utc))
    assert observe(FakeClient(container=c), _svc(), 30) is ObservedFact.BOOTING


def test_running_no_health_past_grace_is_running():
    c = _container("running", started=datetime.now(timezone.utc) - timedelta(seconds=120))
    assert observe(FakeClient(container=c), _svc(), 30) is ObservedFact.RUNNING


def test_exited_zero_is_exited_ok():
    assert observe(FakeClient(container=_container("exited", 0)), _svc(), 30) is ObservedFact.EXITED_OK


def test_exited_nonzero_is_crashed():
    assert observe(FakeClient(container=_container("exited", 137)), _svc(), 30) is ObservedFact.CRASHED
