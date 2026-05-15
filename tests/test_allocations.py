import logging

import pytest

from triton_serve.database.model import Device, Service

LOG = logging.getLogger(pytest.__name__)


@pytest.mark.order(after="test_services.py::test_create_service")
def test_get_resources(test_client, test_db):
    response = test_client.get("/resources")
    LOG.debug(f"response: {response.text}")
    assert response.status_code == 200

    data = response.json()
    assert data["host_name"]
    assert data["num_cpus"] > 0
    assert data["total_memory"] > 0

    cpu = data["cpu"]
    assert cpu["allocated"] >= 0
    assert cpu["in_use"] >= 0
    assert cpu["in_use"] <= cpu["allocated"]

    memory = data["memory"]
    assert memory["allocated"] >= 0
    assert memory["in_use"] >= 0
    assert memory["in_use"] <= memory["allocated"]

    assert isinstance(data["devices"], list)
    for device in data["devices"]:
        assert device["uuid"]
        assert device["index"] >= 0
        assert device["memory"] > 0
        alloc = device["allocation"]
        assert alloc["allocated_pct"] >= 0
        assert alloc["in_use_pct"] >= 0
        assert alloc["in_use_pct"] <= alloc["allocated_pct"]
        assert isinstance(device["services"], list)


@pytest.mark.order(after="test_services.py::test_create_service")
def test_get_resources_totals(test_client, test_db):
    """allocated CPU and memory totals must match the sum across all non-deleted services."""
    services = test_db.query(Service).filter(Service.deleted_at.is_(None)).all()
    expected_cpu = sum(s.resources.cpu_count for s in services if s.resources)
    expected_mem = sum(s.resources.mem_size for s in services if s.resources)

    response = test_client.get("/resources")
    assert response.status_code == 200
    data = response.json()

    assert data["cpu"]["allocated"] == expected_cpu
    assert data["memory"]["allocated"] == expected_mem


@pytest.mark.order(after="test_services.py::test_create_service")
def test_get_resources_service(test_client, test_db):
    service = test_db.query(Service).filter(Service.service_name == "trt-srv_test_svc3").first()
    response = test_client.get(f"/resources/{service.service_id}")
    LOG.debug(f"response: {response.text}")
    assert response.status_code == 200

    data = response.json()
    assert data["service_id"] == service.service_id
    assert data["service_name"] == "trt-srv_test_svc3"
    assert data["cpu_count"] == service.resources.cpu_count
    assert data["mem_size"] == service.resources.mem_size
    assert data["devices"] == []


@pytest.mark.order(after="test_services.py::test_create_service")
def test_get_resources_service_with_gpu(test_client, test_db):
    devices = test_db.query(Device).all()
    if not devices:
        pytest.skip("No GPU devices registered")

    service = test_db.query(Service).filter(Service.service_name == "trt-srv_test_svc1").first()
    response = test_client.get(f"/resources/{service.service_id}")
    LOG.debug(f"response: {response.text}")
    assert response.status_code == 200

    data = response.json()
    assert data["devices"], "expected at least one GPU allocation"
    for dev in data["devices"]:
        assert dev["uuid"]
        assert dev["index"] >= 0
        assert dev["memory"] > 0
        assert 0 < dev["allocation_percentage"] <= 100


@pytest.mark.order(after="test_services.py::test_create_service")
def test_get_resources_service_not_found(test_client):
    response = test_client.get("/resources/-1")
    LOG.debug(f"response: {response.text}")
    assert response.status_code == 404
