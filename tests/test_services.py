import logging

import pytest

LOG = logging.getLogger(pytest.__name__)


@pytest.mark.order(after="model_test.py::test_create_model_success")
@pytest.mark.parametrize(
    "name, models",
    [
        ("trt-srv_test_svc1", ["model_cfg"]),
        ("trt-srv_test_svc2", ["model"]),
        ("trt-srv_test_svc3", ["model_cfg", "model"]),
    ],
)
def test_create_service(test_client, test_docker, name, models):
    response = test_client.post("/services", json={"name": name, "models": models})
    assert response.status_code == 201, f"Cannot create service: {response.json()}"
    # check if container is running
    container = test_docker.containers.get(name)
    assert container.status == "running", f"Container {name} is not created"


@pytest.mark.order(after="test_create_service")
@pytest.mark.parametrize(
    "name, models, expected_status_code",
    [
        ("", ["model_cfg"], 422),
        ("trt-srv_test_svc3", ["nonexistent"], 409),
        ("trt-srv_test_svc4", ["nonexistent"], 409),
        ("trt-srv_test_svc5", ["model_cfg", "nonexistent"], 409),
    ],
)
def test_create_service_wrong_inputs(test_client, name, models, expected_status_code):
    response = test_client.post("/services", json={"name": name, "models": models})
    assert response.status_code == expected_status_code


@pytest.mark.order(after="test_create_service_wrong_inputs")
@pytest.mark.parametrize("name, expected_status_code", [("trt-srv_test_svc1", 204), ("not_existing_service", 404)])
def test_delete_service(test_client, name, expected_status_code):
    response = test_client.delete(f"/services/{name}")
    assert response.status_code == expected_status_code
