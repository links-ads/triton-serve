import logging
import os
from importlib import import_module
from urllib.parse import urlencode

import coloredlogs
import pytest
from fastapi.testclient import TestClient

from src.triton_serve.wsgi import app
from triton_serve.api.schema import ModelSchema

LOG = logging.getLogger(pytest.__name__)
coloredlogs.install(logger=LOG, isatty=True)


# Test with succesful model creatiion using both onnx and config and only the onnx file
@pytest.mark.parametrize(
    "name, version, num_files", [("onnx_and_config", 1, 2), ("only_onnx", 1, 1), ("onnx_and_config", 2, 2)]
)
def test_create_model_success(test_client, test_storage, test_settings, create_zips, name, version, num_files):
    path_to_zip = test_settings.zips_path / (name + ".zip")
    with open(path_to_zip, "rb") as package:
        response = test_client.post(
            "/models",
            data={
                "name": name,
                "version": version,
            },
            files={"package": package},
        )
        assert response.status_code == 201, f"Cannot create model: {response.json()}"
    # check if the model is present inside the models directory, that is /var/serve/models for the test container
    model_path = test_storage.load(ModelSchema(name=name, version=version))
    assert os.path.exists(model_path), "Model not saved in the model repository"
    # check that the files extracted are correct
    filenames = os.listdir(model_path)
    assert len(filenames) == num_files, "Wrong number of files extracted"
    if len(filenames) > 1:
        assert any(f == "config.pbtxt" for f in filenames), "Config file not found"
        assert any(f.endswith(".onnx") for f in filenames), "ONNX file not found"
    else:
        # check that the file is called model.onnx
        assert filenames[0] == "model.onnx", (
            "Missing 'model.onnx' file: either rename it "
            "or set the 'default_model_filename' configuration parameter."
        )


# Test unsuccessful model creation with wrong model zip file
@pytest.mark.parametrize(
    "name, version",
    [
        ("wrong_model_name", 1),
        ("no_model", 1),
    ],
)
def test_create_model_unsuccessful_files(test_client, test_settings, create_zips, name, version):
    path_to_zip = test_settings.zips_path / (name + ".zip")
    with open(path_to_zip, "rb") as package:
        response = test_client.post(
            "/models",
            data={
                "name": name,
                "version": version,
            },
            files={"package": package},
        )
        assert response.status_code == 422


# Test unsuccessful model creation with wrong parameters (name and version))
@pytest.mark.parametrize("name, version", [("", 1), ("model_name", "This_is_not_an_int")])
def test_create_model_unsuccessful_parameters(test_client, test_settings, create_zips, name, version):
    model_name = "onnx_and_config"
    path_to_zip = test_settings.zips_path / (model_name + ".zip")
    with open(path_to_zip, "rb") as package:
        response = test_client.post(
            "/models",
            data={
                "name": name,
                "version": version,
            },
            files={"package": package},
        )
        assert response.status_code == 422


@pytest.mark.order(after="test_create_model_success")
@pytest.mark.parametrize("name, version, num_files", [("onnx_and_config", 1, 2)])
def test_create_model_unsuccess_already_existing(
    test_client, test_storage, test_settings, create_zips, name, version, num_files
):
    path_to_zip = test_settings.zips_path / (name + ".zip")
    with open(path_to_zip, "rb") as package:
        response = test_client.post(
            "/models",
            data={
                "name": name,
                "version": version,
            },
            files={"package": package},
        )
        assert response.status_code == 409, f"Cannot create model: {response.json()}"


@pytest.mark.order(after="test_create_model_success")
@pytest.mark.parametrize(
    "name, version, expected_json",
    [
        (
            "onnx_and_config",
            None,
            [{"name": "onnx_and_config", "version": 1}, {"name": "onnx_and_config", "version": 2}],
        ),
        ("only_onnx", 1, [{"name": "only_onnx", "version": 1}]),
        (
            None,
            None,
            [
                {"name": "onnx_and_config", "version": 1},
                {"name": "only_onnx", "version": 1},
                {"name": "onnx_and_config", "version": 2},
            ],
        ),
        ("not_existent", None, []),
        (None, 1, [{"name": "onnx_and_config", "version": 1}, {"name": "only_onnx", "version": 1}]),
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
    if query_params:
        request_uri += "?" + urlencode(query_params)
    response = test_client.get(request_uri)
    assert response.status_code == 200, f"Cannot get models: {response.json()}"
    assert len(response.json()) == len(expected_json), "Wrong number of models returned"
    # generator expression, for each model in the response generates a boolean value if it is in the expected_json. If all are true the assert is true
    assert all(model in expected_json for model in response.json()), "Wrong models returned"


@pytest.mark.order(after="test_create_model_success")
@pytest.mark.parametrize("name, version", [("", 1), ("onnx_and_config", "this_is_not_an_integer")])
def test_get_models_unsuccessful(test_client, name, version):
    query_params = {}
    if name is not None:
        query_params["model_name"] = name
    if version is not None:
        query_params["version"] = version
    request_uri = "/models"
    if query_params:
        request_uri += "?" + urlencode(query_params)
    response = test_client.get(request_uri)
    assert response.status_code == 422


@pytest.mark.order(after="test_create_model_success")
@pytest.mark.parametrize("name, version", [("onnx_and_config", 1)])
def test_get_model(test_client, name, version):
    response = test_client.get(f"/models/{name}/{version}")
    assert response.status_code == 200, f"Cannot get model: {response.json()}"
    assert response.json() == {"name": name, "version": version}


@pytest.mark.order(after="test_create_model_success")
@pytest.mark.parametrize(
    "name, version, expected_code",
    [
        ("onnx_and_config", 3, 404),
        ("not_existent", 1, 404),
        ("onnx_and_config", -1, 404),
        ("onnx_and_config", "this_is_not_an_int", 422),
        ("onnx_and_config", None, 422),
    ],
)
def test_get_model_unsuccessful(test_client, name, expected_code, version):
    response = test_client.get(f"/models/{name}/{version}")
    assert response.status_code == expected_code, f"This modelo should not exist: {response.json()}"
