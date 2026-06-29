import logging
import time

import pytest
import requests
from docker.errors import NotFound

from triton_serve.database.model import Device, Service, ServiceStatus
from triton_serve.tasks import update_service_status

LOG = logging.getLogger(pytest.__name__)


def test_missing_status_and_retry_fields():
    from triton_serve.config import get_settings
    from triton_serve.database.schema import ServiceSchema

    assert ServiceStatus.MISSING.value == "missing"
    assert "restart_attempts" in ServiceSchema.model_fields
    assert "last_attempt_at" in ServiceSchema.model_fields

    settings = get_settings()
    assert settings.service_max_restart_attempts == 3
    assert settings.service_restart_cooldown == 600


def test_retry_columns_exist(test_db):
    from sqlalchemy import inspect

    cols = {c["name"] for c in inspect(test_db.connection()).get_columns("services")}
    assert {"restart_attempts", "last_attempt_at"} <= cols


@pytest.mark.order(after="test_auth.py::test_api_key_authorized")
@pytest.mark.parametrize(
    "name, models, resources, timeout",
    [
        (
            "trt-srv_test_svc1",
            ["squeezenet"],
            {"gpus": 0.5, "shm_size": 256, "mem_size": 1024},
            3600,
        ),
        (
            "trt-srv_test_svc6",
            ["squeezenet"],
            {"gpus": 0.4, "shm_size": 256, "mem_size": 256},
            3600,
        ),
        (
            "trt-srv_test_svc4",
            ["ensemble_py_step", "ensemble"],
            {"gpus": 0, "shm_size": 256, "mem_size": 1024},
            5,  # set timeout low to test stopping the service
        ),
        (
            "trt-srv_test_svc3",
            ["onnx"],
            {"gpus": 0, "shm_size": 256, "mem_size": 1024},
            3600,
        ),
        (
            "trt-srv_test_svc2",
            ["onnx"],
            {"gpus": 0, "shm_size": 256, "mem_size": 1024},
            3600,
        ),
    ],
)
def test_create_service(test_client, test_docker, test_db, name, models, resources, timeout):
    # get the devices to also check creation when no GPU is available
    devices = set(test_db.query(Device.uuid).all())

    response = test_client.post(
        "/services",
        json={
            "name": name,
            "models": models,
            "resources": resources,
            "timeout": timeout,
        },
    )
    LOG.debug(f"response: {response.text}")
    if resources["gpus"] and not devices:
        assert response.status_code == 409
    else:
        assert response.status_code == 201
        data = response.json()
        assert data["service_name"] == name
        assert data["created_at"] is not None
        # check if container is running
        container = test_docker.containers.get(name)
        assert container.status == "running", f"Container {name} is not created"

        # check if service is in db
        service = test_db.get(Service, ident=data["service_id"])
        assert service.resources is not None
        assert service.resources.shm_size == resources["shm_size"]
        assert service.resources.mem_size == resources["mem_size"]
        assert service.service_name == name
        LOG.debug(f"assigned device: {service.device_allocations}")
        if resources["gpus"]:
            assert service.device_allocations is not None
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

    # verify non-existent service returns 404
    assert test_client.get("/services/-1/config").status_code == 404


@pytest.mark.order(after="test_create_service")
@pytest.mark.parametrize("name,", ["onnx"])
def test_delete_model_in_use(name, test_client, test_settings):
    response = test_client.delete(f"/models/{name}")
    LOG.debug(f"response: {response.text}")
    assert response.status_code == 409
    expected_path = test_settings.repository_path / name
    assert expected_path.exists()


@pytest.mark.order(after="test_create_service")
@pytest.mark.parametrize(
    "service_name, service_container_status",
    [
        ("trt-srv_test_svc4", "exited"),
        ("trt-srv_test_svc2", "running"),
    ],
)
def test_stop_service(
    test_client,
    test_docker,
    test_db,
    service_name,
    service_container_status,
):
    # make sure the service is running
    for _ in range(3):
        container_status = test_docker.containers.get(service_name).status
        LOG.debug(f"container status: {container_status}")
        if container_status == "running":
            break
        LOG.debug(f"{service_name} is not {service_container_status} yet ...")

    # assert that the initial status is indeed "running"
    init_status = test_docker.containers.get(service_name).status
    init_service = test_db.query(Service).filter(Service.service_name == service_name).first()
    LOG.debug(f"service: {init_service}")
    assert init_status == "running"
    assert init_service.container_status in (ServiceStatus.ACTIVE, ServiceStatus.STARTING)
    # make sure we update the service status once the timeout has passed
    time.sleep(5)
    # Manually execute the update service status task: one of the services should be stopped
    # due to the timeout, the other should be running.
    update_service_status.apply(kwargs={"client": test_client})
    assert test_docker.containers.get(service_name).status == service_container_status


@pytest.mark.order(after="test_stop_service")
@pytest.mark.parametrize(
    "service_name, service_container_status, expected_status",
    [
        ("trt-srv_test_svc4", "exited", 204),
        ("trt-srv_test_svc3", "running", 204),
    ],
)
def test_refresh_services(test_docker, test_db, test_client, service_name, service_container_status, expected_status):
    service = test_db.query(Service).filter(Service.service_name == service_name).first()
    initial_srv_status = service.container_status
    service_id = service.service_id if service else -1
    response = test_client.post(f"/services/{service_id}/refresh")
    LOG.debug(f"response: {response.text}")
    assert response.status_code == expected_status
    # check if the container status is the same as the expected status
    assert test_docker.containers.get(service_name).status == service_container_status
    service = test_db.query(Service).filter(Service.service_name == service_name).first()
    assert service.container_status == initial_srv_status


@pytest.mark.order(after="test_refresh_services")
def test_refresh_non_existent(test_client):
    response = test_client.post("/services/-1/refresh")
    LOG.debug(f"response: {response.text}")
    assert response.status_code == 404


@pytest.mark.order(after="test_restart_services")
def test_refresh_force_recreate(test_client, test_docker, test_db):
    """force_recreate should tear down and respawn the container regardless of current state."""
    service = test_db.query(Service).filter(Service.service_name == "trt-srv_test_svc4").first()
    service_id = service.service_id

    response = test_client.post(f"/services/{service_id}/refresh", params={"force_recreate": True})
    LOG.debug(f"response: {response.text}")
    assert response.status_code == 204

    test_db.refresh(service)
    assert service.container_status == ServiceStatus.STARTING
    container = test_docker.containers.get("trt-srv_test_svc4")
    assert container.status == "running"


@pytest.mark.order(after="test_refresh_services")
def test_restart_services(test_docker, test_settings, test_client):
    # make sure it does not work without auth
    response = requests.get("http://traefik/trt-srv_test_svc4/v2/health/ready")
    LOG.debug(f"response: {response.text}")
    assert response.status_code == 403
    # make sure it returns a 404 for a non-existent service
    headers = {"X-API-Key": test_settings.api_keys[0]}
    response = requests.get(
        "http://traefik/whatever/v2/health/ready",
        headers=headers,
    )
    LOG.debug(f"response: {response.text}")
    assert response.status_code == 404
    # we need to 'manually' restart the service since there's no backed running
    test_client.get("status/trt-srv_test_svc4")
    # attempt a couple of times to get a response != 50x using traefik
    for _ in range(3):
        response = requests.get(
            "http://traefik/trt-srv_test_svc4/v2/health/ready",
            headers=headers,
        )
        LOG.debug(f"headers: {response.headers}")
        LOG.debug(f"response: {response.text}")
        if response.status_code not in (502, 503):
            break
        time.sleep(5)

    assert response.status_code == 200
    container = test_docker.containers.get("trt-srv_test_svc4")
    assert container.status == "running"


@pytest.mark.order(after="test_check_service_status")
@pytest.mark.parametrize("name", ["trt-srv_test_svc2", "trt-srv_test_svc3"])
def test_triton_ping_unathorized(name):
    url = f"http://traefik/{name}/v2/health/ready"
    response = requests.get(url)
    # try three times to get a response, with a timeout of 60 seconds
    for _ in range(3):
        response = requests.get(url, timeout=60)
        LOG.debug(f"response: {response.text}")
        if response.status_code != 404:
            break
        else:
            time.sleep(5)

    assert response.status_code == 403
    data = response.json()
    assert "Invalid API Key" in data["message"]


@pytest.mark.order(after="test_triton_ping_unathorized")
@pytest.mark.parametrize("name", ["trt-srv_test_svc2", "trt-srv_test_svc3"])
def test_triton_ping(name, test_settings):
    url = f"http://traefik/{name}/v2/health/ready"
    headers = {"X-API-Key": test_settings.api_keys[0]}
    # try three times to get a response, with a timeout of 60 seconds
    response = None
    for _ in range(3):
        response = requests.get(url, timeout=60, headers=headers)
        LOG.debug(f"response: {response.text}")
        if response.status_code == 200:
            break
        else:
            time.sleep(5)
    assert response and response.status_code == 200


@pytest.mark.order(after="test_triton_ping")
@pytest.mark.parametrize("name, model", [("trt-srv_test_svc2", "onnx")])
def test_triton_models_ready(name, model, test_settings):
    url = f"http://traefik/{name}/v2/models/{model}/ready"
    headers = {"X-API-Key": test_settings.api_keys[0]}

    # try three times to get a response, with a timeout of 60 seconds
    response = None
    for _ in range(3):
        response = requests.get(url, timeout=60, headers=headers)
        if response.status_code == 200:
            break
        else:
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


@pytest.mark.order(after="test_update_service")
def test_update_service_recreate(test_client, test_docker, test_db):
    service = test_db.query(Service).filter(Service.service_name == "trt-srv_test_svc2").first()
    service_id = service.service_id
    old_container_id = service.container_id

    response = test_client.put(f"/services/{service_id}", params={"recreate": True}, json={"timeout": 5400})
    LOG.debug(f"response: {response.text}")
    assert response.status_code == 200

    data = response.json()
    assert data["inactivity_timeout"] == 5400
    test_db.refresh(service)
    assert service.inactivity_timeout == 5400
    assert service.container_status == ServiceStatus.STARTING
    assert service.container_id != old_container_id
    container = test_docker.containers.get("trt-srv_test_svc2")
    assert container.status == "running"


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


@pytest.mark.order(after="test_update_service_recreate")
def test_check_status_missing_is_observational(test_client, test_docker, test_db):
    """A vanished container becomes MISSING, never DELETED, and is not auto-recreated by a read."""
    service = test_db.query(Service).filter(Service.service_name == "trt-srv_test_svc3").first()
    service_id = service.service_id

    # remove the container out-of-band
    test_docker.containers.get("trt-srv_test_svc3").remove(force=True)

    # a plain read must observe MISSING without spawning anything or deleting the service
    response = test_client.get(f"/services/{service_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["container_status"] == ServiceStatus.MISSING.value
    assert data["deleted_at"] is None
    with pytest.raises(NotFound):
        test_docker.containers.get("trt-srv_test_svc3")

    # restore for the rest of the suite
    test_client.post(f"/services/{service_id}/refresh", params={"force_recreate": True})


@pytest.mark.order(after="test_check_status_missing_is_observational")
def test_missing_recreated_same_id_new_container(test_client, test_docker, test_db):
    """A MISSING service is recreated with the same service_id and a fresh container_id."""
    service = test_db.query(Service).filter(Service.service_name == "trt-srv_test_svc3").first()
    service_id = service.service_id
    old_container_id = service.container_id

    test_docker.containers.get("trt-srv_test_svc3").remove(force=True)
    # mark MISSING via an observational read
    test_client.get(f"/services/{service_id}")

    response = test_client.post(f"/services/{service_id}/refresh")
    assert response.status_code == 204

    test_db.refresh(service)
    assert service.service_id == service_id
    assert service.container_status == ServiceStatus.STARTING
    assert service.container_id != old_container_id
    assert test_docker.containers.get("trt-srv_test_svc3").status in ("running", "created")


@pytest.mark.order(after="test_missing_recreated_same_id_new_container")
def test_missing_recreate_no_double_spawn(test_docker, test_db, test_settings):
    """Re-verify guard: reconcile on a MISSING service whose container is actually present must not spawn a new one."""
    from triton_serve.api.services import domain

    service = test_db.query(Service).filter(Service.service_name == "trt-srv_test_svc3").first()
    healthy_container_id = service.container_id
    # force MISSING while the container is genuinely still present
    service.container_status = ServiceStatus.MISSING
    test_db.commit()

    domain.reconcile_missing_container(
        db=test_db,
        client=test_docker,
        service=service,
        service_network=test_settings.service_network,
        service_models_volume=test_settings.service_volume,
        max_restart_attempts=test_settings.service_max_restart_attempts,
        restart_cooldown=test_settings.service_restart_cooldown,
    )

    test_db.refresh(service)
    # the re-verify guard saw the container present -> no new container, no extra attempt
    assert service.container_id == healthy_container_id
    assert service.container_status != ServiceStatus.MISSING


@pytest.mark.order(after="test_missing_recreate_no_double_spawn")
def test_missing_exhaustion_to_error_then_recovery(test_client, test_docker, test_db):
    """Repeated recreate failures land the service in ERROR, recoverable via force_recreate."""
    service = test_db.query(Service).filter(Service.service_name == "trt-srv_test_svc3").first()
    service_id = service.service_id
    good_image = service.service_image

    # force every recreate attempt to fail by pointing at a bogus image
    service.service_image = "ghcr.io/links-ads/does-not-exist:0"
    service.container_status = ServiceStatus.MISSING
    service.restart_attempts = 0
    service.last_attempt_at = None
    test_db.commit()
    try:
        test_docker.containers.get("trt-srv_test_svc3").remove(force=True)
    except NotFound:
        pass

    # drive past the budget: max_restart_attempts (3) spawn attempts + 1 observation -> ERROR
    for _ in range(5):
        test_client.post(f"/services/{service_id}/refresh")
    test_db.refresh(service)
    assert service.container_status == ServiceStatus.ERROR

    # recover: restore a valid image and force_recreate
    service.service_image = good_image
    test_db.commit()
    response = test_client.post(f"/services/{service_id}/refresh", params={"force_recreate": True})
    assert response.status_code == 204
    test_db.refresh(service)
    assert service.container_status == ServiceStatus.STARTING

    # leave a clean retry budget so downstream tests are not affected by exhaustion
    service.restart_attempts = 0
    service.last_attempt_at = None
    test_db.commit()


@pytest.mark.order(after="test_missing_exhaustion_to_error_then_recovery")
def test_status_endpoint_recreates_missing(test_client, test_docker, test_db):
    """GET /status on a MISSING service triggers recreation and returns 202."""
    service = test_db.query(Service).filter(Service.service_name == "trt-srv_test_svc3").first()
    service_id = service.service_id

    test_docker.containers.get("trt-srv_test_svc3").remove(force=True)
    test_client.get(f"/services/{service_id}")  # observe -> MISSING

    response = test_client.get("/status/trt-srv_test_svc3")
    assert response.status_code == 200  # decorator status; body carries the signal
    assert response.json() == 202

    test_db.refresh(service)
    assert service.container_status == ServiceStatus.STARTING
    assert test_docker.containers.get("trt-srv_test_svc3").status in ("running", "created")


@pytest.mark.order(after="test_status_endpoint_recreates_missing")
def test_sentinel_reconciles_missing(test_client, test_docker, test_db):
    """The sentinel task recreates a MISSING container via /refresh."""
    service = test_db.query(Service).filter(Service.service_name == "trt-srv_test_svc3").first()
    service_id = service.service_id
    old_container_id = service.container_id

    test_docker.containers.get("trt-srv_test_svc3").remove(force=True)
    test_client.get(f"/services/{service_id}")  # observe -> MISSING
    test_db.refresh(service)
    assert service.container_status == ServiceStatus.MISSING

    update_service_status.apply(kwargs={"client": test_client})

    test_db.refresh(service)
    assert service.container_status == ServiceStatus.STARTING
    assert service.container_id != old_container_id


@pytest.mark.order(after="test_update_service_recreate")
def test_delete_services(test_client, test_docker, test_db):
    # get all services
    services = test_db.query(Service).all()
    for service in services:
        query_params = {"delete_container": True}
        response = test_client.delete(f"/services/{service.service_id}", params=query_params)
        data = response.json()
        assert response.status_code == 202
        assert data["service_id"] == service.service_id
        assert data["deleted_at"] is not None
        # check if container has been deleted
        with pytest.raises(NotFound):
            test_docker.containers.get(service.container_id)
