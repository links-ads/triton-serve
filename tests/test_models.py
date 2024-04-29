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


@pytest.mark.parametrize(
    "model",
    ["empty", "no_config", "no_model", "no_platform", "wrong_name", "wrong_version", "wrong_extension"],
)
def test_create_models_from_zip_wrong_content(test_client, make_zip, model):
    """
    Test with unsuccessful model creation when using a wrong model name, or no model at all.
    """
    with make_zip(include_models=[model]) as src:
        response = test_client.post("/models", files={"package": src})
    LOG.debug(f"Response: {response.json()}")
    assert response.status_code == 422


@pytest.mark.order(after="test_get_models_empty")
@pytest.mark.parametrize("model", ["ensemble", "onnx", "python"])
def test_create_models_from_zip(test_client, test_settings, make_zip, model):
    """
    Test with successful model creation using both onnx and config and only the onnx file.
    """
    with make_zip(include_models=[model]) as src:
        response = test_client.post("/models", files={"package": src})
    LOG.debug(f"Response: {response.json()}")
    assert response.status_code == 201
    # check if the model is present inside the models repository
    expected_model_root = test_settings.repository_path / model
    assert expected_model_root.exists() and expected_model_root.is_dir()
    # check that the files extracted are correct
    assert (expected_model_root / "config.pbtxt").exists()
    # gather any subdirectories
    # check that the version folders are present and contain files
    subdirs = [f for f in expected_model_root.iterdir() if f.is_dir()]
    for subdir in subdirs:
        assert subdir.name.isdigit()
        files = [f for f in subdir.iterdir() if f.is_file()]
        assert len(files) > 0


@pytest.mark.order(after="test_create_models_from_zip")
@pytest.mark.parametrize("model", [("ensemble")])
def test_create_models_from_zip_already_existing(test_client, make_zip, model):
    with make_zip(include_models=[model]) as package:
        response = test_client.post("/models", files={"package": package})
    LOG.debug(f"Response: {response.json()}")
    assert response.status_code == 409


@pytest.mark.order(after="test_create_models_from_zip")
@pytest.mark.parametrize("model", ["ensemble", "onnx"])
def test_create_models_from_zip_already_existing_with_update(test_client, test_settings, make_zip, model):
    with make_zip(include_models=[model]) as package:
        response = test_client.post("/models", data={"update": True}, files={"package": package})
    LOG.debug(f"Response: {response.json()}")
    assert response.status_code == 201
    # check if the model is present inside the models repository
    expected_model_root = test_settings.repository_path / model
    assert expected_model_root.exists() and expected_model_root.is_dir()
    # check that the files extracted are correct
    assert (expected_model_root / "config.pbtxt").exists()
    # gather any subdirectories
    # check that the version folders are present and contain files
    subdirs = [f for f in expected_model_root.iterdir() if f.is_dir()]
    for subdir in subdirs:
        assert subdir.name.isdigit()
        files = [f for f in subdir.iterdir() if f.is_file()]
        assert len(files) > 0


@pytest.mark.order(after="test_create_models_from_zip")
def test_create_models_from_repo(test_client, test_settings, test_repository):
    # gather existing models first
    repository_root = test_settings.repository_path
    model_dirs = {d for d in repository_root.iterdir() if d.is_dir()}
    # create models from the repository
    response = test_client.post("/models/repository", params={"repository_url": test_repository})
    LOG.debug(f"Response: {response.json()}")
    assert response.status_code == 201
    # check if the models are present inside the models repository
    updated_model_dirs = {d for d in repository_root.iterdir() if d.is_dir()}
    diff = updated_model_dirs - model_dirs
    assert len(diff) > 0
    for model_dir in diff:
        assert (model_dir / "config.pbtxt").exists()
        # gather any subdirectories
        # check that the version folders are present and contain files
        subdirs = [f for f in model_dir.iterdir() if f.is_dir()]
        for subdir in subdirs:
            assert subdir.name.isdigit()
            files = [f for f in subdir.iterdir() if f.is_file()]
            assert len(files) > 0


@pytest.mark.order(after="test_create_models_from_zip_already_existing")
@pytest.mark.parametrize(
    "name, version, count",
    [
        ("ensemble", None, 2),
        ("onnx", 1, 1),
        (None, None, 5),
        ("non_existent", None, 0),
        (None, 1, 4),
    ],
)
def test_get_models(test_client, name, version, count):
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
    assert len(data) == count
    for item in data:
        assert "model_name" in item
        assert "model_version" in item
        assert "model_type" in item
        assert "model_uri" in item
        assert "created_at" in item
        assert "updated_at" in item
        assert "source" in item
        if name is not None:
            assert item["model_name"] == name
        if version is not None:
            assert item["model_version"] == version
        assert item["model_type"] in ["ensemble", "onnx", "python"]


@pytest.mark.order(after="test_create_models_from_zip_already_existing")
@pytest.mark.parametrize("name, version", [("", 1), ("whatever", "not_an_integer")])
def test_get_models_wrong_params(test_client, name, version):
    query_params = {}
    if name is not None:
        query_params["model_name"] = name
    if version is not None:
        query_params["version"] = version
    request_uri = "/models"
    response = test_client.get(request_uri, params=query_params)
    assert response.status_code == 422


@pytest.mark.order(after="test_create_models_from_zip_already_existing")
@pytest.mark.parametrize("name, version", [("onnx", 1), ("python", 1)])
def test_get_model(test_client, name, version):
    response = test_client.get(f"/models/{name}/{version}")
    assert response.status_code == 200
    data = response.json()
    assert data["model_name"] == name
    assert data["model_version"] == version


@pytest.mark.order(after="test_get_models")
@pytest.mark.parametrize(
    "name, version, expected_code",
    [
        ("ensemble", 3, 404),
        ("non_existent", 1, 404),
        ("onnx", -1, 404),
        ("onnx", "this_is_not_an_int", 422),
        ("onnx", None, 422),
    ],
)
def test_get_model_wrong_params(test_client, name, expected_code, version):
    response = test_client.get(f"/models/{name}/{version}")
    assert response.status_code == expected_code


@pytest.mark.order(after="test_get_models_wrong_params")
@pytest.mark.parametrize("name, version", [("non_existent", 1)])
def test_delete_non_existent_model(name, version, test_client):
    response = test_client.delete(f"/models/{name}/{version}")
    assert response.status_code == 404


@pytest.mark.order(after="test_get_models_wrong_params")
@pytest.mark.parametrize("name, version", [("python", 1)])
def test_delete_model(name, version, test_client, test_settings):
    response = test_client.delete(f"/models/{name}/{version}")
    assert response.status_code == 204
    expected_path = test_settings.repository_path / name / str(version)
    assert not expected_path.exists()
