import logging
import time

import pytest
import requests
from docker.errors import NotFound

from triton_serve.database.model import Device, Service

LOG = logging.getLogger(pytest.__name__)


@pytest.mark.order(after="test_auth.py::test_api_key_authorized")
@pytest.mark.parametrize(
    "name, models, resources",
    [
        ("trt-srv_test_svc1", [{"name": "ensemble", "version": 2}], {"gpus": 1, "shm_size": 256, "mem_size": 4096}),
        ("trt-srv_test_svc3", [{"name": "ensemble", "version": 2}], {"gpus": 0, "shm_size": 256, "mem_size": 4096}),
        ("trt-srv_test_svc2", [{"name": "onnx", "version": 1}], {"gpus": 0, "shm_size": 256, "mem_size": 4096}),
    ],
)
def test_create_service(test_client, test_docker, test_db, name, models, resources):
    # get the devices to also check creation when no GPU is available
    devices = set(test_db.query(Device.uuid).all())

    response = test_client.post("/services", json={"name": name, "models": models, "resources": resources})
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
    for _ in range(3):
        response = requests.get(url, timeout=60, headers=headers)
        LOG.debug(f"response: {response.text}")
        if response.status_code == 200:
            break
        else:
            time.sleep(5)
    assert response.status_code == 200


@pytest.mark.order(after="test_triton_ping")
@pytest.mark.parametrize("name, model", [("trt-srv_test_svc2", "onnx")])
def test_triton_models_ready(name, model, test_settings):
    url = f"http://traefik/{name}/v2/models/{model}/ready"
    headers = {"X-API-Key": test_settings.api_keys[0]}

    # try three times to get a response, with a timeout of 60 seconds
    for _ in range(3):
        response = requests.get(url, timeout=60, headers=headers)
        if response.status_code == 200:
            break
        else:
            time.sleep(5)
    assert response.status_code == 200


@pytest.mark.order(after="test_triton_models_ready")
@pytest.mark.parametrize(
    "name, models, expected_status_code",
    [
        ("", ["ensemble"], 422),
        ("trt-srv_test_svc3", [{"name": "nonexistent", "version": 1}], 409),
        ("trt-srv_test_svc4", [{"name": "nonexistent", "version": 1}], 409),
        ("trt-srv_test_svc5", [{"name": "nonexistent", "version": 1}], 409),
    ],
)
def test_create_service_wrong_inputs(test_client, name, models, expected_status_code):
    response = test_client.post("/services", json={"name": name, "models": models})
    assert response.status_code == expected_status_code


@pytest.mark.order(after="test_create_service_wrong_inputs")
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
