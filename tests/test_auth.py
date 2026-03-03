import logging
from datetime import datetime, timedelta, timezone

import pytest

from triton_serve.api.dto import APIKeyCreateBody, ServiceKeyCreateBody
from triton_serve.database.model import APIKey, KeyType

LOG = logging.getLogger(pytest.__name__)


@pytest.fixture
def create_api_key(test_db):
    def _create_api_key(key_type, key_value, project, notes=None, expiration_days=30):
        expires_at = datetime.now(tz=timezone.utc) + timedelta(days=expiration_days)

        api_key = APIKey(
            key_type=key_type,
            value="test_key_" + key_type.value + key_value,
            project=project,
            notes=notes,
            expires_at=expires_at,
        )
        test_db.add(api_key)
        test_db.commit()
        return api_key

    return _create_api_key


@pytest.mark.order(after="test_models.py::test_create_models_from_zip")
def test_api_key_authorized(test_client, test_settings):
    # make a get request for /models and set the X-API-Key header to the app_secret
    response = test_client.get("/models", headers={"X-API-Key": test_settings.api_keys[0]})
    assert response.status_code == 200


@pytest.mark.order(after="test_api_key_authorized")
def test_api_key_unauthorized(test_client):
    # make a get request for /models and set the X-API-Key header to the app_secret
    response = test_client.get("/models", headers={"X-API-Key": "invalid"})
    assert response.status_code == 401


@pytest.mark.parametrize("key_type", [KeyType.USER, KeyType.ADMIN])
def test_create_api_key(test_client, key_type):
    key_data = APIKeyCreateBody(project="test_project", key_type=key_type, notes="Test key", expiration_days=30)
    response = test_client.post("/keys", json=key_data.model_dump(mode="json"))
    assert response.status_code == 201
    data = response.json()
    assert data["key_type"] == key_type.value
    assert data["project"] == "test_project"
    assert data["notes"] == "Test key"
    assert "value" in data
    assert "expires_at" in data


@pytest.mark.order(after="test_services.py::test_delete_services")
def test_create_service_key(test_client):
    # First, create a service
    service_response = test_client.post(
        "/services",
        json={
            "name": "trt-srv_test_test_service",
            "models": ["ensemble"],
            "resources": {"gpus": 0, "shm_size": 256, "mem_size": 4096},
        },
    )
    assert service_response.status_code == 201
    service_id = service_response.json()["service_id"]

    # Now create a service key
    key_data = ServiceKeyCreateBody(project="test_project", notes="Test service key", expiration_days=30)
    response = test_client.post(f"/keys/{service_id}", json=key_data.model_dump(mode="json"))
    assert response.status_code == 201
    data = response.json()
    assert data["key_type"] == KeyType.SERVICE.value
    assert data["project"] == "test_project"
    assert data["notes"] == "Test service key"
    assert "value" in data
    assert "expires_at" in data
    assert len(data["services"]) == 1
    assert data["services"][0]["service_id"] == service_id


@pytest.mark.order(after="test_api_key_unauthorized")
def test_list_api_keys(test_client, create_api_key):
    create_api_key(KeyType.USER, "key1", "project1")
    create_api_key(KeyType.ADMIN, "key2", "project2")
    create_api_key(KeyType.SERVICE, "key3", "project3")

    response = test_client.get("/keys")
    assert response.status_code == 200
    data = response.json()
    LOG.debug(data)

    assert len(data) >= 3
    assert "project1" in [key["project"] for key in data]
    assert "project2" in [key["project"] for key in data]
    assert "project3" in [key["project"] for key in data]
    assert KeyType.ADMIN.value in [key["key_type"] for key in data]
    assert KeyType.USER.value in [key["key_type"] for key in data]
    assert KeyType.SERVICE.value in [key["key_type"] for key in data]

    # Test filtering by key_type
    response = test_client.get("/keys", params={"key_type": KeyType.USER.value})
    assert response.status_code == 200
    data = response.json()
    assert len(data) > 0
    assert data[0]["key_type"] == KeyType.USER.value

    # Test filtering by project
    response = test_client.get("/keys", params={"project": "project2"})
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 1
    assert data[0]["project"] == "project2"


@pytest.mark.order(after="test_list_api_keys")
def test_update_api_key(test_client, create_api_key):
    api_key = create_api_key(KeyType.USER, "key_old", "old_project")
    update_data = {"project": "new_project", "notes": "Updated notes"}
    LOG.debug(f"Updating key {api_key.value}")
    response = test_client.put(f"/keys/{api_key.value}", json=update_data)
    assert response.status_code == 200
    data = response.json()
    LOG.debug(f"Updated key: {data}")
    assert data["project"] == "new_project"
    assert data["notes"] == "Updated notes"


@pytest.mark.order(after="test_update_api_key")
def test_revoke_api_key(test_client, create_api_key, test_db):
    api_key = create_api_key(KeyType.USER, "deleteme", "test_project")

    response = test_client.delete(f"/keys/{api_key.value}")
    LOG.debug(f"Response: {response.text}")
    assert response.status_code == 204

    # Verify the key has been deleted
    deleted_key = test_db.query(APIKey).filter(APIKey.value == api_key.value).first()
    assert deleted_key is None


@pytest.mark.order(after="test_revoke_api_key")
def test_add_service_to_key(
    test_client,
    create_api_key,
):
    api_key = create_api_key(KeyType.SERVICE, "svc1", "status_check")

    # Create a service
    service_response = test_client.post(
        "/services",
        json={
            "name": "trt-srv_test_another_test_service",
            "models": ["onnx"],
            "resources": {"gpus": 0, "shm_size": 256, "mem_size": 4096},
        },
    )
    LOG.debug("Service response: %s", service_response.text)
    assert service_response.status_code == 201
    service_id = service_response.json()["service_id"]

    # Add service to key
    response = test_client.post(f"/keys/{api_key.key_id}/services/{service_id}")
    LOG.debug("Add service response: %s", response.text)
    assert response.status_code == 200
    data = response.json()
    assert len(data["services"]) == 1
    assert data["services"][0]["service_id"] == service_id

    retry = test_client.post(f"/keys/{api_key.key_id}/services/{service_id}")
    assert retry.status_code == 400


@pytest.mark.order(after="test_add_service_to_key")
def test_check_service_status(test_client):
    # using the client, get the service key associated with test project
    response = test_client.get("/keys", params={"project": "status_check"})
    LOG.debug("Response: %s", response.text)
    assert response.status_code == 200
    data = response.json()
    assert len(data) > 0
    key = data[0]["value"]

    # Test with user key
    response = test_client.get(
        "/status/trt-srv_test_another_test_service",
        headers={"X-API-Key": "test_key_userkey1"},
    )
    LOG.debug(response.text)
    assert response.status_code == 403

    # Test with no key
    response = test_client.get(
        "/status/trt-srv_test_another_test_service",
        headers={"X-API-Key": ""},
    )
    LOG.debug(response.text)
    assert response.status_code == 401

    # Test with invalid key
    response = test_client.get(
        "/status/trt-srv_test_another_test_service",
        headers={"X-API-Key": "invalid_key"},
    )
    LOG.debug(response.text)
    assert response.status_code == 401

    # test with the service key
    response = test_client.get(
        "/status/trt-srv_test_another_test_service",
        headers={"X-API-Key": key},
    )
    LOG.debug(response.text)
    assert response.status_code == 200


@pytest.mark.order(after="test_add_service_to_key")
def test_remove_service_from_key(test_client, create_api_key):
    api_key = create_api_key(KeyType.SERVICE, "svc2", "test_project")

    # Create a service and add it to the key
    service_response = test_client.post(
        "/services",
        json={
            "name": "trt-srv_yet_another_test_service",
            "models": ["onnx"],
            "resources": {"gpus": 0, "shm_size": 256, "mem_size": 4096},
        },
    )
    assert service_response.status_code == 201
    service_id = service_response.json()["service_id"]

    test_client.post(f"/keys/{api_key.key_id}/services/{service_id}")

    # Remove service from key
    response = test_client.delete(f"/keys/{api_key.key_id}/services/{service_id}")
    assert response.status_code == 204


@pytest.mark.order(after="test_revoke_api_key")
def test_create_api_key_invalid_type(test_client):
    key_data = APIKeyCreateBody(
        project="test_project",
        key_type=KeyType.SERVICE,  # This should fail
        notes="Test key",
        expiration_days=30,
    )
    response = test_client.post("/keys", json=key_data.model_dump(mode="json"))
    assert response.status_code == 400


@pytest.mark.order(after="test_revoke_api_key")
def test_create_service_key_nonexistent_service(test_client):
    key_data = ServiceKeyCreateBody(project="test_project", notes="Test service key", expiration_days=30)
    response = test_client.post("/keys/99999", json=key_data.model_dump(mode="json"))
    assert response.status_code == 404


@pytest.mark.order(after="test_revoke_api_key")
def test_update_nonexistent_key(test_client):
    update_data = {"project": "new_project", "notes": "Updated notes"}
    response = test_client.put("/keys/nonexistent_key", json=update_data)
    assert response.status_code == 404


@pytest.mark.order(after="test_revoke_api_key")
def test_revoke_nonexistent_key(test_client):
    response = test_client.delete("/keys/nonexistent_key")
    assert response.status_code == 404


@pytest.mark.order(after="test_revoke_api_key")
def test_add_service_to_non_service_key(test_client, create_api_key):
    api_key1 = create_api_key(KeyType.USER, "usr", "test_project")
    api_key2 = create_api_key(KeyType.ADMIN, "adm", "test_project")

    response = test_client.post(f"/keys/{api_key1.key_id}/services/1")
    assert response.status_code == 400
    response = test_client.post(f"/keys/{api_key2.key_id}/services/1")
    assert response.status_code == 400


@pytest.mark.order(after="test_revoke_api_key")
def test_remove_service_from_non_service_key(test_client, create_api_key):
    api_key1 = create_api_key(KeyType.USER, "usr2", "test_project")
    api_key2 = create_api_key(KeyType.ADMIN, "adm2", "test_project")

    response = test_client.delete(f"/keys/{api_key1.key_id}/services/1")
    assert response.status_code == 400
    response = test_client.delete(f"/keys/{api_key2.key_id}/services/1")
    assert response.status_code == 400
