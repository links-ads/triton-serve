import logging

import pytest

LOG = logging.getLogger(pytest.__name__)


def test_get_models_empty(test_client):
    """
    Test with empty models repository.
    """
    response = test_client.get("/models")
    assert response.status_code == 200
    assert response.json() == []


@pytest.mark.order(after="test_get_models_empty")
@pytest.mark.parametrize("name, version", [("model_cfg", 1), ("model", 1), ("model_cfg", 2)])
def test_create_model(test_client, test_settings, make_zip, name, version):
    """
    Test with successful model creation using both onnx and config and only the onnx file.
    """
    has_config = "cfg" in name
    with make_zip(add_model=True, add_config=has_config) as src:
        metadata = {"name": name, "version": version}
        response = test_client.post("/models", data=metadata, files={"package": src})
    LOG.debug(f"Response: {response.json()}")
    assert response.status_code == 201
    # check if the model is present inside the models repository
    expected_config_path = test_settings.repository_path / name
    expected_model_path = expected_config_path / str(version)
    assert expected_model_path.exists() and expected_model_path.is_dir() and expected_model_path.is_dir()
    # check that the files extracted are correct
    cfg_files = [f for f in expected_config_path.iterdir() if f.is_file()]
    model_files = [f for f in expected_model_path.iterdir() if f.is_file()]
    len(model_files) == 1
    if has_config:
        assert len(cfg_files) == 1
        assert cfg_files[0].name == "config.pbtxt"
    assert model_files[0].name == "model.onnx"


@pytest.mark.parametrize(
    "name, version",
    [
        ("whatever", 1),
        ("missing", 1),
    ],
)
def test_create_model_wrong_files(test_client, make_zip, name, version):
    """
    Test with unsuccessful model creation when using a wrong model name, or no model at all.
    """
    add_model = name != "missing"  # config files only are not allowed
    add_config = name != "whatever"  # custom names require a config file
    with make_zip(model_name=f"{name}.onnx", add_model=add_model, add_config=add_config) as src:
        response = test_client.post("/models", data={"name": name, "version": version}, files={"package": src})
    LOG.debug(f"Response: {response.json()}")
    assert response.status_code == 422


@pytest.mark.parametrize("name, version", [("", 1), ("model_name", "not_an_int")])
def test_create_model_wrong_params(test_client, make_zip, name, version):
    """
    Test with unsuccessful model creation when an empty name is passed or a wrong version.
    """
    with make_zip() as src:
        response = test_client.post("/models", data={"name": name, "version": version}, files={"package": src})
    LOG.debug(f"Response: {response.json()}")
    assert response.status_code == 422


@pytest.mark.order(after="test_create_model")
@pytest.mark.parametrize("name, version", [("model_cfg", 1)])
def test_create_model_already_existing(test_client, make_zip, name, version):
    with make_zip(add_model=True, add_config=True) as package:
        response = test_client.post("/models", data={"name": name, "version": version}, files={"package": package})
    LOG.debug(f"Response: {response.json()}")
    assert response.status_code == 409


@pytest.mark.order(after="test_create_model_already_existing")
@pytest.mark.parametrize(
    "name, version, expected_json",
    [
        (
            "model_cfg",
            None,
            [
                {
                    "model_name": "model_cfg",
                    "model_version": 1,
                    "model_uri": "/var/serve/models/model_cfg/1",
                    "description": None,
                },
                {
                    "model_name": "model_cfg",
                    "model_version": 2,
                    "model_uri": "/var/serve/models/model_cfg/2",
                    "description": None,
                },
            ],
        ),
        (
            "model",
            1,
            [
                {
                    "model_name": "model",
                    "model_version": 1,
                    "model_uri": "/var/serve/models/model/1",
                    "description": None,
                }
            ],
        ),
        (
            None,
            None,
            [
                {
                    "model_name": "model_cfg",
                    "model_version": 1,
                    "model_uri": "/var/serve/models/model_cfg/1",
                    "description": None,
                },
                {
                    "model_name": "model",
                    "model_version": 1,
                    "model_uri": "/var/serve/models/model/1",
                    "description": None,
                },
                {
                    "model_name": "model_cfg",
                    "model_version": 2,
                    "model_uri": "/var/serve/models/model_cfg/2",
                    "description": None,
                },
            ],
        ),
        ("not_existent", None, []),
        (
            None,
            1,
            [
                {
                    "model_name": "model_cfg",
                    "model_version": 1,
                    "model_uri": "/var/serve/models/model_cfg/1",
                    "description": None,
                },
                {
                    "model_name": "model",
                    "model_version": 1,
                    "model_uri": "/var/serve/models/model/1",
                    "description": None,
                },
            ],
        ),
    ],
)
def test_get_models(test_client, name, version, expected_json):
    # build the request uri with the query parameters
    query_params = {}
    if name is not None:
        query_params["model_name"] = name
    if version is not None:
        query_params["version"] = version
    request_uri = "/models"
    response = test_client.get(request_uri, params=query_params)
    assert response.status_code == 200
    data = response.json()
    LOG.debug(f"Params: {query_params}")
    LOG.debug(f"Response: {data}")
    assert len(data) == len(expected_json)
    for expected, actual in zip(expected_json, data):
        for key, value in expected.items():
            assert actual[key] == value


@pytest.mark.order(after="test_create_model_already_existing")
@pytest.mark.parametrize("name, version", [("", 1), ("model_cfg", "not_an_integer")])
def test_get_models_wrong_params(test_client, name, version):
    query_params = {}
    if name is not None:
        query_params["model_name"] = name
    if version is not None:
        query_params["version"] = version
    request_uri = "/models"
    response = test_client.get(request_uri, params=query_params)
    assert response.status_code == 422


@pytest.mark.order(after="test_create_model_already_existing")
@pytest.mark.parametrize("name, version", [("model_cfg", 1), ("model", 1)])
def test_get_model(test_client, name, version):
    response = test_client.get(f"/models/{name}/{version}")
    assert response.status_code == 200
    data = response.json()
    assert data["model_name"] == name
    assert data["model_version"] == version


@pytest.mark.order(after="test_get_model")
@pytest.mark.parametrize(
    "name, version, expected_code",
    [
        ("model_cfg", 3, 404),
        ("not_existent", 1, 404),
        ("model_cfg", -1, 404),
        ("model_cfg", "this_is_not_an_int", 422),
        ("model_cfg", None, 422),
    ],
)
def test_get_model_wrong_params(test_client, name, expected_code, version):
    response = test_client.get(f"/models/{name}/{version}")
    assert response.status_code == expected_code


@pytest.mark.order(after="test_get_model_wrong_params")
@pytest.mark.parametrize("name, version", [("not_existing", 1)])
def test_delete_non_existent_model(name, version, test_client):
    response = test_client.delete(f"/models/{name}/{version}")
    assert response.status_code == 404


@pytest.mark.order(after="test_get_model_wrong_params")
@pytest.mark.parametrize("name, version", [("model_cfg", 1)])
def test_delete_model(name, version, test_client, test_settings):
    response = test_client.delete(f"/models/{name}/{version}")
    assert response.status_code == 204
    expected_path = test_settings.repository_path / name / str(version)
    assert not expected_path.exists()
