import logging
import time
from datetime import UTC

import pytest
import requests

from triton_serve.database.model import DesiredState, Device, Model, RuntimeStatus, Service
from triton_serve.tasks import update_service_status

LOG = logging.getLogger(pytest.__name__)


def _drive_reconciler(test_db, service, until, ticks=24, delay=5):
    """Manually tick the reconciler (no celery beat in test mode) until `service` reaches a
    target runtime_status, refreshing the ORM object between ticks. Returns the final status."""
    for _ in range(ticks):
        update_service_status.apply()
        test_db.refresh(service)
        if service.runtime_status in until:
            break
        time.sleep(delay)
    return service.runtime_status


def test_lifecycle_fields_and_settings():
    from triton_serve.config import get_settings
    from triton_serve.database.schema import ServiceSchema

    assert "restart_attempts" in ServiceSchema.model_fields
    assert "last_attempt_at" in ServiceSchema.model_fields
    assert "runtime_status" in ServiceSchema.model_fields
    assert "desired_state" in ServiceSchema.model_fields

    settings = get_settings()
    assert settings.service_max_restart_attempts == 3
    assert settings.service_restart_cooldown == 600


def test_retry_columns_exist(test_db):
    from sqlalchemy import inspect

    cols = {c["name"] for c in inspect(test_db.connection()).get_columns("services")}
    assert {"restart_attempts", "last_attempt_at"} <= cols


def test_lifecycle_enums_and_columns(test_db):
    from sqlalchemy import inspect

    assert {s.value for s in DesiredState} == {"available", "suspended", "retired"}
    assert {s.value for s in RuntimeStatus} == {
        "ready",
        "warming",
        "idle",
        "recovering",
        "failed",
        "suspended",
        "retired",
    }
    cols = {c["name"] for c in inspect(test_db.connection()).get_columns("services")}
    assert {"desired_state", "runtime_status"} <= cols


@pytest.mark.order(after="test_auth.py::test_api_key_authorized")
@pytest.mark.parametrize(
    "name, models, resources, timeout",
    [
        ("trt-srv_test_svc1", ["squeezenet"], {"gpus": 0.5, "shm_size": 256, "mem_size": 1024}, 3600),
        ("trt-srv_test_svc6", ["squeezenet"], {"gpus": 0.4, "shm_size": 256, "mem_size": 256}, 3600),
        ("trt-srv_test_svc4", ["ensemble_py_step", "ensemble"], {"gpus": 0, "shm_size": 256, "mem_size": 1024}, 5),
        ("trt-srv_test_svc3", ["onnx"], {"gpus": 0, "shm_size": 256, "mem_size": 1024}, 3600),
        ("trt-srv_test_svc2", ["onnx"], {"gpus": 0, "shm_size": 256, "mem_size": 1024}, 3600),
    ],
)
def test_create_service(test_client, test_db, name, models, resources, timeout):
    """Create is declarative: a 201 record with runtime_status=WARMING and no container yet."""
    devices = set(test_db.query(Device.uuid).all())

    response = test_client.post(
        "/services",
        json={"name": name, "models": models, "resources": resources, "timeout": timeout},
    )
    LOG.debug(f"response: {response.text}")
    if resources["gpus"] and not devices:
        assert response.status_code == 409
        return

    assert response.status_code == 201
    data = response.json()
    assert data["service_name"] == name
    assert data["created_at"] is not None
    # declarative contract: warming intent, container spawned by the reconciler out of band
    assert data["runtime_status"] == RuntimeStatus.WARMING.value
    assert data["desired_state"] == DesiredState.AVAILABLE.value
    assert data["container_id"] is None

    service = test_db.get(Service, ident=data["service_id"])
    assert service.resources is not None
    assert service.resources.shm_size == resources["shm_size"]
    assert service.resources.mem_size == resources["mem_size"]
    assert service.service_name == name
    if resources["gpus"]:
        assert service.device_allocations
    else:
        assert not service.device_allocations


@pytest.mark.order(after="test_create_service")
def test_get_service_config(test_client, test_db):
    service = test_db.query(Service).filter(Service.service_name == "trt-srv_test_svc3").first()
    response = test_client.get(f"/services/{service.service_id}/config")
    LOG.debug(f"response: {response.text}")
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "trt-srv_test_svc3"
    assert data["models"] == ["onnx"]
    assert data["timeout"] == 3600
    assert data["resources"]["shm_size"] == 256
    assert data["resources"]["mem_size"] == 1024
    assert data["resources"]["gpus"] == 0.0

    assert test_client.get("/services/-1/config").status_code == 404


@pytest.mark.order(after="test_create_service")
@pytest.mark.parametrize("name,", ["onnx"])
def test_delete_model_in_use(name, test_client, test_settings):
    response = test_client.delete(f"/models/{name}")
    LOG.debug(f"response: {response.text}")
    assert response.status_code == 409
    expected_path = test_settings.repository_path / name
    assert expected_path.exists()


@pytest.mark.order(after="test_get_service_config")
def test_reconciler_brings_service_ready(test_db, test_docker):
    """The reconciler spawns and readies a declaratively-created service on its ticks."""
    service = test_db.query(Service).filter(Service.service_name == "trt-srv_test_svc2").one()
    status = _drive_reconciler(test_db, service, until={RuntimeStatus.READY}, ticks=36, delay=5)
    assert status == RuntimeStatus.READY, f"service did not become READY, stuck at {status}"
    container = test_docker.containers.get("trt-srv_test_svc2")
    assert container.status == "running"


@pytest.mark.order(after="test_reconciler_brings_service_ready")
@pytest.mark.parametrize("name", ["trt-srv_test_svc2"])
def test_triton_ping_unauthorized(name):
    url = f"http://traefik/{name}/v2/health/ready"
    response = requests.get(url)
    for _ in range(3):
        response = requests.get(url, timeout=60)
        LOG.debug(f"response: {response.text}")
        if response.status_code != 404:
            break
        time.sleep(5)
    assert response.status_code == 403
    assert "Invalid API Key" in response.json()["message"]


@pytest.mark.order(after="test_triton_ping_unauthorized")
@pytest.mark.parametrize("name", ["trt-srv_test_svc2"])
def test_triton_ping(name, test_settings):
    url = f"http://traefik/{name}/v2/health/ready"
    headers = {"X-API-Key": test_settings.api_keys[0]}
    response = None
    for _ in range(6):
        response = requests.get(url, timeout=60, headers=headers)
        LOG.debug(f"response: {response.text}")
        if response.status_code == 200:
            break
        time.sleep(5)
    assert response and response.status_code == 200


@pytest.mark.order(after="test_triton_ping")
@pytest.mark.parametrize("name, model", [("trt-srv_test_svc2", "onnx")])
def test_triton_models_ready(name, model, test_settings):
    url = f"http://traefik/{name}/v2/models/{model}/ready"
    headers = {"X-API-Key": test_settings.api_keys[0]}
    response = None
    for _ in range(6):
        response = requests.get(url, timeout=60, headers=headers)
        if response.status_code == 200:
            break
        time.sleep(5)
    assert response and response.status_code == 200


@pytest.mark.order(after="test_triton_models_ready")
@pytest.mark.parametrize(
    "name, models, expected_status_code",
    [
        ("", ["ensemble"], 422),
        ("trt-srv_test_svc4", [""], 422),
        ("trt-srv_test_svc5", ["nonexistent"], 409),
    ],
)
def test_create_service_wrong_inputs(test_client, name, models, expected_status_code):
    response = test_client.post("/services", json={"name": name, "models": models})
    LOG.debug(f"response: {response.text}")
    assert response.status_code == expected_status_code


@pytest.mark.order(after="test_create_service_wrong_inputs")
def test_update_service(test_client, test_db):
    service = test_db.query(Service).filter(Service.service_name == "trt-srv_test_svc3").first()
    service_id = service.service_id

    response = test_client.put(f"/services/{service_id}", json={"timeout": 7200, "priority": 2})
    LOG.debug(f"response: {response.text}")
    assert response.status_code == 200

    data = response.json()
    assert data["inactivity_timeout"] == 7200
    assert data["priority"] == 2
    test_db.refresh(service)
    assert service.inactivity_timeout == 7200
    assert service.priority == 2


@pytest.mark.order(after="test_create_service_wrong_inputs")
@pytest.mark.parametrize(
    "service_id, update_body, expected_status",
    [
        (-1, {"timeout": 100}, 404),
        (-1, {"models": []}, 422),
    ],
)
def test_update_service_wrong_inputs(test_client, service_id, update_body, expected_status):
    response = test_client.put(f"/services/{service_id}", json=update_body)
    LOG.debug(f"response: {response.text}")
    assert response.status_code == expected_status


@pytest.mark.order(after="test_update_service")
def test_status_projection_codes(test_client, test_db, test_settings):
    svc = test_db.query(Service).filter(Service.service_name == "trt-srv_test_svc3").one()

    svc.desired_state, svc.runtime_status = DesiredState.AVAILABLE, RuntimeStatus.READY
    test_db.commit()
    assert test_client.get("/status/trt-srv_test_svc3").status_code == 200

    svc.runtime_status = RuntimeStatus.IDLE
    test_db.commit()
    r = test_client.get("/status/trt-srv_test_svc3")
    # Retry-After tracks the reconcile tick so a client polls about once per bring-up opportunity
    assert r.status_code == 503 and r.headers.get("Retry-After") == str(test_settings.sentinel_poll_interval)

    svc.desired_state, svc.runtime_status = DesiredState.SUSPENDED, RuntimeStatus.SUSPENDED
    test_db.commit()
    r = test_client.get("/status/trt-srv_test_svc3")
    assert r.status_code == 503 and "Retry-After" not in r.headers

    svc.runtime_status = RuntimeStatus.RETIRED
    test_db.commit()
    assert test_client.get("/status/trt-srv_test_svc3").status_code == 404

    # restore so downstream tests see an available service
    svc.desired_state, svc.runtime_status = DesiredState.AVAILABLE, RuntimeStatus.READY
    test_db.commit()


@pytest.mark.order(after="test_status_projection_codes")
def test_suspend_and_resume_set_intent(test_client, test_db):
    svc = test_db.query(Service).filter(Service.service_name == "trt-srv_test_svc3").one()

    assert test_client.post(f"/services/{svc.service_id}/suspend").status_code == 204
    test_db.refresh(svc)
    assert svc.desired_state == DesiredState.SUSPENDED

    assert test_client.post(f"/services/{svc.service_id}/resume").status_code == 204
    test_db.refresh(svc)
    assert svc.desired_state == DesiredState.AVAILABLE


@pytest.mark.order(after="test_suspend_and_resume_set_intent")
def test_retry_resets_budget_and_recovers(test_client, test_db):
    svc = test_db.query(Service).filter(Service.service_name == "trt-srv_test_svc3").one()
    svc.restart_attempts = 3
    svc.runtime_status = RuntimeStatus.FAILED
    test_db.commit()

    assert test_client.post(f"/services/{svc.service_id}/retry").status_code == 204
    test_db.refresh(svc)
    assert svc.restart_attempts == 0
    assert svc.last_attempt_at is None
    assert svc.runtime_status == RuntimeStatus.RECOVERING


@pytest.mark.order(after="test_retry_resets_budget_and_recovers")
def test_execute_recreates_absent(test_db, test_docker, test_settings):
    from triton_serve.api.services.execute import execute
    from triton_serve.api.services.reconcile import Action, Decision

    service = test_db.query(Service).filter(Service.service_name == "trt-srv_test_svc2").one()
    # simulate a vanished container: point the record at a non-existent id
    service.container_id = "deadbeefdead"
    test_db.commit()

    execute(
        db=test_db,
        client=test_docker,
        service=service,
        decision=Decision(Action.RECREATE, RuntimeStatus.WARMING),
        settings=test_settings,
    )
    test_db.refresh(service)
    assert service.runtime_status == RuntimeStatus.WARMING
    assert test_docker.containers.get(service.service_name) is not None


@pytest.mark.order(after="test_execute_recreates_absent")
def test_reconciler_idles_then_wakes(test_db, test_docker, test_settings):
    """The reconciler scales an idle service to zero, then wakes it once it sees traffic again."""
    from datetime import datetime

    svc = test_db.query(Service).filter(Service.service_name == "trt-srv_test_svc4").one()  # timeout=5s
    svc.desired_state = DesiredState.AVAILABLE
    test_db.commit()
    time.sleep(6)  # exceed the 5s inactivity window
    update_service_status.apply()
    test_db.refresh(svc)
    assert svc.runtime_status == RuntimeStatus.IDLE  # scaled to zero, container stopped

    svc.last_active_time = datetime.now(UTC)
    test_db.commit()
    # target back to 1 -> the reconciler must actually bring it up; IDLE would mean it never woke
    status = _drive_reconciler(test_db, svc, until={RuntimeStatus.WARMING, RuntimeStatus.READY}, ticks=6, delay=2)
    assert status in (RuntimeStatus.WARMING, RuntimeStatus.READY), f"service did not wake, stuck at {status}"


@pytest.mark.order(after="test_reconciler_idles_then_wakes")
def test_delete_is_db_only(test_client, test_db):
    """Delete is DB-only: it tombstones the record and drops it from listings; the reconciler
    tears down the container out of band."""
    services = test_db.query(Service).filter(Service.deleted_at.is_(None)).all()
    for service in services:
        service_id = service.service_id
        response = test_client.delete(f"/services/{service_id}")
        assert response.status_code == 204
        test_db.refresh(service)
        assert service.deleted_at is not None
        assert service.desired_state == DesiredState.RETIRED

    listing = test_client.get("/services").json()
    assert listing == []


@pytest.mark.order(after="test_delete_is_db_only")
def test_bad_image_service_ends_failed(test_db, test_docker, test_settings):
    """Spec headline guarantee: a bad image ref surfaces asynchronously as FAILED within the crash
    budget, and the reconciler does not spin forever recreating (the C1 regression). Self-contained
    (direct DB insert, no models) and ordered last so it cannot disturb the ordered lifecycle chain."""
    from datetime import datetime, timedelta

    from docker.errors import NotFound

    name = "trt-srv_test_badimg"
    svc = Service(
        service_name=name,
        service_image="ghcr.io/links-ads/does-not-exist:0",
        priority=1,
        last_active_time=datetime.now(UTC),
        desired_state=DesiredState.AVAILABLE,
        runtime_status=RuntimeStatus.WARMING,
    )
    test_db.add(svc)
    test_db.commit()

    # C1: a failed pull spends the budget instead of looping at attempts=0 forever
    update_service_status.apply()
    test_db.refresh(svc)
    assert svc.restart_attempts >= 1, "failed pull did not advance the crash budget"

    # exhaust the budget (bypass the real backoff wait) and confirm it lands terminal in FAILED
    svc.restart_attempts = test_settings.service_max_restart_attempts
    svc.last_attempt_at = datetime.now(UTC) - timedelta(seconds=120)
    test_db.commit()
    status = _drive_reconciler(test_db, svc, until={RuntimeStatus.FAILED}, ticks=4, delay=2)
    assert status == RuntimeStatus.FAILED, f"bad-image service did not reach FAILED, stuck at {status}"
    with pytest.raises(NotFound):
        test_docker.containers.get(name)


@pytest.mark.order(after="test_bad_image_service_ends_failed")
def test_failed_stays_terminal_after_cooldown(test_db, test_settings):
    """FAILED is terminal: elapsed time must NOT resurrect a failed service by silently resetting
    its crash budget. Before the fix, any FAILED service revived one cooldown after its last attempt;
    only POST /retry may reset the budget now."""
    from datetime import datetime, timedelta

    svc = test_db.query(Service).filter(Service.service_name == "trt-srv_test_badimg").one()
    assert svc.runtime_status == RuntimeStatus.FAILED
    # push the last attempt well past the reset window -- the old time-based reset would revive it
    svc.last_attempt_at = datetime.now(UTC) - timedelta(seconds=test_settings.service_restart_cooldown + 60)
    test_db.commit()

    update_service_status.apply()
    test_db.refresh(svc)
    assert svc.runtime_status == RuntimeStatus.FAILED, "an elapsed cooldown resurrected a FAILED service"
    assert svc.restart_attempts == test_settings.service_max_restart_attempts, "crash budget was silently reset"


@pytest.mark.order(after="test_failed_stays_terminal_after_cooldown")
def test_budget_returns_after_sustained_ready(test_db, test_docker, test_settings):
    """A recovered service that stays READY past the cooldown earns its crash budget back, but a
    brief READY within the cooldown does not (so a fast crash-recover-crash flap cannot loop)."""
    from datetime import datetime, timedelta

    from triton_serve.api.services.execute import execute
    from triton_serve.api.services.reconcile import Action, Decision

    svc = test_db.query(Service).filter(Service.service_name == "trt-srv_test_badimg").one()
    ready = Decision(Action.NONE, RuntimeStatus.READY)

    # within the cooldown: a fresh recovery attempt has not yet proven stable -> keep the budget spent
    svc.restart_attempts = 2
    svc.last_attempt_at = datetime.now(UTC) - timedelta(seconds=5)
    svc.runtime_status = RuntimeStatus.RECOVERING
    test_db.commit()
    execute(db=test_db, client=test_docker, service=svc, decision=ready, settings=test_settings)
    test_db.refresh(svc)
    assert svc.restart_attempts == 2

    # past the cooldown: sustained health returns the budget
    svc.last_attempt_at = datetime.now(UTC) - timedelta(seconds=test_settings.service_restart_cooldown + 60)
    test_db.commit()
    execute(db=test_db, client=test_docker, service=svc, decision=ready, settings=test_settings)
    test_db.refresh(svc)
    assert svc.restart_attempts == 0
    assert svc.last_attempt_at is None
    assert svc.runtime_status == RuntimeStatus.READY


RESOURCES = {"gpus": 0, "shm_size": 256, "mem_size": 1024}


@pytest.mark.order(after="test_create_service")
def test_created_service_gets_an_image_row(test_client, test_db):
    response = test_client.post(
        "/services",
        json={"name": "trt-srv_test_img1", "models": ["onnx"], "resources": RESOURCES, "timeout": 3600},
    )
    assert response.status_code == 201, response.text
    service_id = response.json()["service_id"]
    try:
        test_db.expire_all()
        service = test_db.get(Service, ident=service_id)
        assert service.image is not None
        assert service.image_hash == service.image.image_hash
        assert service.image.base_image == service.service_image
    finally:
        test_client.delete(f"/services/{service_id}")


@pytest.mark.order(after="test_create_service")
def test_invalid_dependency_is_rejected_at_write_time(test_client, test_db):
    model = test_db.query(Model).filter(Model.model_name == "onnx").one()
    original = list(model.dependencies or [])
    model.dependencies = ["--extra-index-url http://attacker.example"]
    test_db.commit()
    try:
        response = test_client.post(
            "/services",
            json={"name": "trt-srv_test_img2", "models": ["onnx"], "resources": RESOURCES, "timeout": 3600},
        )
        assert response.status_code == 422, response.text
    finally:
        model.dependencies = original
        test_db.commit()


def test_retry_requeues_a_build_stuck_building(test_db, test_settings, monkeypatch):
    """A worker that dies mid-build leaves the row BUILDING with no task behind it."""
    from datetime import datetime

    from triton_serve.api.services import domain
    from triton_serve.builder.resolve import image_from_spec
    from triton_serve.builder.spec import make_build_spec
    from triton_serve.database.model import ImageStatus, ServiceImage

    spec = make_build_spec(
        base_image="ghcr.io/links-ads/serve-triton:23.07-py3", apt_packages=[], pip_packages=["six==1.16.1"]
    )
    image = image_from_spec(spec, test_settings)
    image.status = ImageStatus.BUILDING
    image.build_log = "worker died here"
    service = Service(
        service_name="trt-srv_test_stuck",
        service_image=spec.base_image,
        last_active_time=datetime.now(UTC),
        priority=1,
        image=image,
    )
    test_db.add(service)
    test_db.commit()

    enqueued: list[str] = []
    monkeypatch.setattr(domain, "enqueue_build", enqueued.append)
    try:
        domain.reset_and_wake(test_db, service.service_id)
        test_db.refresh(image)
        assert image.status is ImageStatus.PENDING
        assert image.build_log is None
        assert enqueued == [image.image_hash]
    finally:
        test_db.delete(service)
        test_db.commit()
        test_db.query(ServiceImage).filter(ServiceImage.image_hash == spec.image_hash).delete()
        test_db.commit()
