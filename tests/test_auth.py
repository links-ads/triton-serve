import logging
from datetime import UTC, datetime, timedelta

import pytest

from triton_serve.api.dto import APIKeyCreateBody, ServiceKeyCreateBody
from triton_serve.database.model import APIKey, DesiredState, KeyType, RuntimeStatus, Service

LOG = logging.getLogger(pytest.__name__)


@pytest.fixture
def create_api_key(test_db):
    def _create_api_key(key_type, key_value, project, notes=None, expiration_days=30):
        expires_at = datetime.now(tz=UTC) + timedelta(days=expiration_days)

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


@pytest.fixture
def status_of(test_client, test_db):
    def _status_of(service_name: str, key: str):
        # READY so an entitled key projects to 200: the request path itself never spawns a container
        service = test_db.query(Service).filter(Service.service_name == service_name).one()
        service.runtime_status = RuntimeStatus.READY
        test_db.commit()
        return test_client.get(f"/status/{service_name}", headers={"X-API-Key": key})

    return _status_of


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


@pytest.mark.order(after="test_services.py::test_delete_is_db_only")
def test_create_service_key(test_client, status_of):
    service_name = "trt-srv_test_test_service"
    # First, create a service
    service_response = test_client.post(
        "/services",
        json={
            "name": service_name,
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

    # the new key must reach the service it was created for
    assert status_of(service_name, data["value"]).status_code == 200


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
    status_of,
):
    api_key = create_api_key(KeyType.SERVICE, "svc1", "status_check")
    service_name = "trt-srv_test_another_test_service"

    # Create a service
    service_response = test_client.post(
        "/services",
        json={
            "name": service_name,
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

    # associating an existing key must let it reach the service (issue 105)
    assert status_of(service_name, api_key.value).status_code == 200

    retry = test_client.post(f"/keys/{api_key.key_id}/services/{service_id}")
    assert retry.status_code == 400


@pytest.mark.order(after="test_add_service_to_key")
def test_status_endpoint_auth(test_client, status_of):
    service_name = "trt-srv_test_another_test_service"

    # using the client, get the service key associated with test project
    response = test_client.get("/keys", params={"project": "status_check"})
    LOG.debug("Response: %s", response.text)
    assert response.status_code == 200
    data = response.json()
    assert len(data) > 0
    key = data[0]["value"]

    assert status_of(service_name, "test_key_userkey1").status_code == 403
    assert status_of(service_name, "").status_code == 401
    assert status_of(service_name, "invalid_key").status_code == 401
    assert status_of(service_name, key).status_code == 200


@pytest.mark.order(after="test_add_service_to_key")
def test_remove_service_from_key(test_client, create_api_key, status_of):
    api_key = create_api_key(KeyType.SERVICE, "svc2", "test_project")
    service_name = "trt-srv_yet_another_test_service"

    # Create a service and add it to the key
    service_response = test_client.post(
        "/services",
        json={
            "name": service_name,
            "models": ["onnx"],
            "resources": {"gpus": 0, "shm_size": 256, "mem_size": 4096},
        },
    )
    assert service_response.status_code == 201
    service_id = service_response.json()["service_id"]

    test_client.post(f"/keys/{api_key.key_id}/services/{service_id}")
    assert status_of(service_name, api_key.value).status_code == 200

    # Remove service from key
    response = test_client.delete(f"/keys/{api_key.key_id}/services/{service_id}")
    assert response.status_code == 204
    assert status_of(service_name, api_key.value).status_code == 403


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
def test_create_service_key_deleted_service(test_client, test_db):
    """A tombstoned service counts as absent: keying it would rebuild Traefik routing for a
    service the reconciler has already torn down."""
    svc = Service(
        service_name="trt-srv_test_keyed_tombstone",
        service_image="ghcr.io/links-ads/does-not-exist:0",
        priority=1,
        last_active_time=datetime.now(UTC),
        desired_state=DesiredState.RETIRED,
        runtime_status=RuntimeStatus.RETIRED,
        deleted_at=datetime.now(UTC),
    )
    test_db.add(svc)
    test_db.commit()

    key_data = ServiceKeyCreateBody(project="test_project", notes="Test service key", expiration_days=30)
    response = test_client.post(f"/keys/{svc.service_id}", json=key_data.model_dump(mode="json"))
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


@pytest.mark.order(after="test_status_endpoint_auth")
def test_status_rejects_a_service_key_for_another_service(create_api_key, status_of):
    """A SERVICE key reaches only the services it is associated with."""
    outsider = create_api_key(KeyType.SERVICE, "outsider", "scoping")

    assert status_of("trt-srv_test_another_test_service", outsider.value).status_code == 403


@pytest.mark.order(after="test_status_endpoint_auth")
def test_status_accepts_the_master_admin_key_for_any_service(test_settings, status_of):
    """Master keys are seeded as ADMIN rows by the populate migration and reach everything."""
    assert status_of("trt-srv_test_another_test_service", test_settings.api_keys[0]).status_code == 200


@pytest.mark.order(after="test_status_endpoint_auth")
def test_status_does_not_record_wake_intent_for_an_unentitled_key(test_client, create_api_key, test_db):
    """403 must precede the status match: every branch of it records wake intent."""
    outsider = create_api_key(KeyType.SERVICE, "nowake", "scoping")
    service = test_db.query(Service).filter(Service.service_name == "trt-srv_test_another_test_service").one()
    service.runtime_status = RuntimeStatus.IDLE
    test_db.commit()
    before = service.last_active_time

    response = test_client.get(
        "/status/trt-srv_test_another_test_service",
        headers={"X-API-Key": outsider.value},
    )

    assert response.status_code == 403
    test_db.refresh(service)
    assert service.last_active_time == before


@pytest.mark.order(after="test_status_endpoint_auth")
def test_status_does_not_rewrite_a_fresh_liveness_timestamp(test_client, test_db, test_settings):
    """The reconciler cannot observe sub-window precision, so a fresh timestamp is left alone."""
    service = test_db.query(Service).filter(Service.service_name == "trt-srv_test_another_test_service").one()
    service.runtime_status = RuntimeStatus.READY
    service.inactivity_timeout = 3600  # a 36s coalescing window
    service.last_active_time = datetime.now(UTC)
    test_db.commit()
    before = service.last_active_time

    response = test_client.get(
        "/status/trt-srv_test_another_test_service",
        headers={"X-API-Key": test_settings.api_keys[0]},
    )

    assert response.status_code == 200
    test_db.refresh(service)
    assert service.last_active_time == before


@pytest.mark.order(after="test_status_endpoint_auth")
def test_status_writes_liveness_once_the_window_has_elapsed(test_client, test_db, test_settings):
    """Past the window the write must land, or the reconciler would scale a live service to zero."""
    service = test_db.query(Service).filter(Service.service_name == "trt-srv_test_another_test_service").one()
    service.runtime_status = RuntimeStatus.READY
    service.inactivity_timeout = 3600
    service.last_active_time = datetime.now(UTC) - timedelta(seconds=40)
    test_db.commit()
    before = service.last_active_time

    response = test_client.get(
        "/status/trt-srv_test_another_test_service",
        headers={"X-API-Key": test_settings.api_keys[0]},
    )

    assert response.status_code == 200
    test_db.refresh(service)
    assert service.last_active_time > before


@pytest.mark.order(after="test_status_endpoint_auth")
def test_status_always_writes_liveness_for_a_short_inactivity_timeout(test_client, test_db, test_settings):
    """A timeout below the divisor yields a zero window: coalescing disables itself rather than
    letting a service be stopped while it is serving. Pinned so nobody adds a floor."""
    service = test_db.query(Service).filter(Service.service_name == "trt-srv_test_another_test_service").one()
    service.runtime_status = RuntimeStatus.READY
    service.inactivity_timeout = 5  # 5 // 100 == 0
    service.last_active_time = datetime.now(UTC)
    test_db.commit()
    before = service.last_active_time

    try:
        response = test_client.get(
            "/status/trt-srv_test_another_test_service",
            headers={"X-API-Key": test_settings.api_keys[0]},
        )

        # the reconciler may have moved the service off READY under a 5s timeout; either branch
        # records activity, so the timestamp is the assertion that matters here
        assert response.status_code in (200, 503)
        test_db.refresh(service)
        assert service.last_active_time > before
    finally:
        service.inactivity_timeout = 3600
        service.last_active_time = datetime.now(UTC)
        test_db.commit()


@pytest.mark.order(after="test_status_endpoint_auth")
def test_status_still_records_wake_intent_for_an_idle_service(test_client, test_db, test_settings):
    """Scale-from-zero depends on the IDLE branch recording intent; coalescing must not break it."""
    service = test_db.query(Service).filter(Service.service_name == "trt-srv_test_another_test_service").one()
    service.runtime_status = RuntimeStatus.IDLE
    service.inactivity_timeout = 3600
    service.last_active_time = datetime.now(UTC) - timedelta(seconds=40)
    test_db.commit()
    before = service.last_active_time

    response = test_client.get(
        "/status/trt-srv_test_another_test_service",
        headers={"X-API-Key": test_settings.api_keys[0]},
    )

    assert response.status_code == 503
    assert "Retry-After" in response.headers
    test_db.refresh(service)
    assert service.last_active_time > before


@pytest.mark.order(after="test_status_endpoint_auth")
def test_status_reports_404_before_403_for_an_unknown_service(test_client, create_api_key):
    """Absence outranks entitlement. The single-statement lookup must still return the row
    regardless of entitlement, or this collapses into a 403."""
    outsider = create_api_key(KeyType.SERVICE, "ordering", "scoping")

    response = test_client.get(
        "/status/trt-srv_test_no_such_service",
        headers={"X-API-Key": outsider.value},
    )

    assert response.status_code == 404
