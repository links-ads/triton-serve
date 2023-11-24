import logging
import os
from importlib import import_module
from urllib.parse import urlencode

import coloredlogs
import docker
import pytest

LOG = logging.getLogger(pytest.__name__)
coloredlogs.install(logger=LOG, isatty=True)


# Test service creation
@pytest.mark.order(after="test_create_model_success")
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
    response = test_client.post("/services", json={"name": name, "models": models})
    assert response.status_code == 201, f"Cannot create service: {response.json()}"
    # check if container is running
    container = docker_client.containers.get(name)

    assert container.attrs["State"] == "running", f"Container {name} is not running"
    spawned_containers_names.append(name)
