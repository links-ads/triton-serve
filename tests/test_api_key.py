import logging

import pytest

LOG = logging.getLogger(pytest.__name__)


@pytest.mark.order(after="test_models.py::test_get_model_wrong_params")
def test_api_key_authorized(test_client, test_settings):
    # make a get request for /models and set the X-API-Key header to the app_secret
    response = test_client.get("/models", headers={"X-API-Key": test_settings.api_keys[0]})
    assert response.status_code == 200


@pytest.mark.order(after="test_api_key_authorized")
def test_api_key_unauthorized(test_client):
    # make a get request for /models and set the X-API-Key header to the app_secret
    response = test_client.get("/models", headers={"X-API-Key": "invalid"})
    assert response.status_code == 401
