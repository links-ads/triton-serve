import logging
import os
from importlib import import_module
from time import sleep
from urllib.parse import urlencode

import coloredlogs
import docker
import pytest

LOG = logging.getLogger(pytest.__name__)
coloredlogs.install(logger=LOG, isatty=True)


# Test service creation
@pytest.mark.order(after="model_test.py::test_create_model_success")
@pytest.mark.parametrize(
    "name, models",
    [
        ("test_service_1", ["onnx_and_config"]),
        ("test_service_2", ["only_onnx"]),
        ("test_service_3", ["onnx_and_config", "only_onnx"]),
    ],
)
def test_create_service_success(test_client, test_containers, name, models):
    docker_client = test_containers["client"]
    spawned_containers_names = test_containers["spawned_containers_names"]
    spawned_containers_names.append(name)
    response = test_client.post("/services", json={"name": name, "models": models})
    assert response.status_code == 201, f"Cannot create service: {response.json()}"
    # check if container is running
    container = docker_client.containers.get(name)

    assert container.status == "running", f"Container {name} is not created"


# Test service creation unsuccessful
@pytest.mark.order(after="test_create_service_success")
@pytest.mark.parametrize(
    "name, models, expected_status_code",
    [
        ("", ["onnx_and_config"], 422),
        ("test_service_3", ["this_model_does_not_exist"], 409),
        ("test_service_4", ["this_model_does_not_exist"], 409),
        ("test_service_5", ["onnx_and_config", "this_model_does_not_exist"], 409),
    ],
)
def test_create_service_unsuccessful(test_client, test_containers, name, models, expected_status_code):
    spawned_containers_names = test_containers["spawned_containers_names"]
    spawned_containers_names.append(name)
    response = test_client.post("/services", json={"name": name, "models": models})
    assert response.status_code == expected_status_code, f"Cannot create service: {response.json()}"


@pytest.mark.order(after="test_create_service_success")
@pytest.mark.parametrize("name, expected_status_code", [("test_service_1", 204), ("not_existing_service", 404)])
def test_delete_model(test_client, name, expected_status_code):
    response = test_client.delete(f"/services/{name}")
    assert response.status_code == expected_status_code, f"Cannot delete service: {response.json()}"
