import io
import logging
import os
from pathlib import Path
from typing import cast

import pytest

from triton_serve.storage.validation import parse_dependencies

LOG = logging.getLogger(pytest.__name__)
DATA = Path(__file__).parent / "data"


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
    LOG.debug(f"Response: {response.text}")
    assert response.status_code == 422


@pytest.mark.order(after="test_get_models_empty")
@pytest.mark.parametrize("model", ["ensemble", "ensemble_py_step", "onnx", "python"])
def test_create_models_from_zip(test_client, test_settings, make_zip, model):
    """
    Test with successful model creation using both onnx and config and only the onnx file.
    """
    with make_zip(include_models=[model]) as src:
        src = cast(io.BytesIO, src)
        response = test_client.post("/models", files={"package": src})
    LOG.info(f"Response: {response.text}")
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
@pytest.mark.parametrize("model", ["ensemble"])
def test_create_models_from_zip_already_existing(test_client, make_zip, model):
    with make_zip(include_models=[model]) as package:
        response = test_client.post("/models", files={"package": package})
    LOG.debug(f"Response: {response.text}")
    assert response.status_code == 409


@pytest.mark.order(after="test_create_models_from_zip")
@pytest.mark.parametrize("model", ["ensemble", "onnx"])
def test_create_models_from_zip_already_existing_with_update(test_client, test_settings, make_zip, model):
    with make_zip(include_models=[model]) as package:
        response = test_client.post("/models", data={"update": True}, files={"package": package})
    LOG.debug(f"Response: {response.text}")
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
def test_create_models_from_repo_wrong_url(test_client):
    response = test_client.post("/models/repository", params={"repository_url": "https://github.com"})
    LOG.debug(f"Response: {response.text}")
    assert response.status_code == 400


@pytest.mark.order(after="test_create_models_from_zip")
def test_create_models_from_repo(test_client, test_settings, test_repository):
    repository_root = test_settings.repository_path
    model_dirs = {d for d in repository_root.iterdir() if d.is_dir()}

    response = test_client.post("/models/repository", params={"repository_url": test_repository})
    LOG.debug(f"Response: {response.text}")
    assert response.status_code == 201

    updated_model_dirs = {d for d in repository_root.iterdir() if d.is_dir()}
    diff = updated_model_dirs - model_dirs
    assert len(diff) > 0

    for model_dir in diff:
        assert (model_dir / "config.pbtxt").exists()
        subdirs = [f for f in model_dir.iterdir() if f.is_dir()]
        for subdir in subdirs:
            assert subdir.name.isdigit()
            files = [f for f in subdir.iterdir() if f.is_file()]
            assert len(files) > 0

            for file in files:
                # Check if file is not empty
                assert os.path.getsize(file) > 0, f"File {file} is empty"

                # Check if file is not a Git LFS pointer file
                with open(file, "rb") as f:
                    content = f.read(100)  # Read first 100 bytes
                    is_text = True
                    for byte in content:
                        if byte < 32 and byte != 9 and byte != 10 and byte != 13:
                            is_text = False
                            break

                if is_text:
                    with open(file) as f:
                        first_line = f.readline().strip()
                        assert not first_line.startswith("version https://git-lfs.github.com/spec/"), (
                            f"File {file} appears to be a Git LFS pointer"
                        )

                # If the file is large (> 1MB), check that it's not just filled with the same byte
                if os.path.getsize(file) > 1_000_000:
                    with open(file, "rb") as f:
                        first_byte = f.read(1)
                        f.seek(1000)  # Check at 1000 bytes in
                        middle_byte = f.read(1)
                        f.seek(-1, 2)  # Check last byte
                        last_byte = f.read(1)
                    assert not (first_byte == middle_byte == last_byte), (
                        f"Large file {file} appears to be filled with the same byte"
                    )

    LOG.debug(f"Successfully verified {len(diff)} new model directories")


@pytest.mark.order(after="test_create_models_from_zip_already_existing")
@pytest.mark.parametrize(
    "name, source, count",
    [
        ("ensemble", None, 1),
        ("onnx", None, 1),
        ("non_existent", None, 0),
        (None, None, 5),
        (None, "test_repository", 1),
        (None, "test_archive", 4),
        ("onnx", "test_archive", 1),
    ],
)
def test_get_models(test_client, test_repository, test_archive, name, source, count):
    # build the request uri with the query parameters
    query_params = {}
    if name is not None:
        query_params["model_name"] = name
    if source is not None:
        query_params["source"] = test_archive if source == "test_archive" else test_repository

    request_uri = "/models"
    response = test_client.get(request_uri, params=query_params)
    assert response.status_code == 200
    data = response.json()
    LOG.debug(f"Params: {query_params}")
    LOG.debug(f"Response: {data}")
    assert len(data) == count
    for item in data:
        assert "model_name" in item
        assert "model_type" in item
        assert "created_at" in item
        assert "updated_at" in item
        assert "source" in item
        if name is not None:
            assert item["model_name"] == name
        assert item["model_type"] in ["ensemble", "onnxruntime_onnx", "python"]
        assert "versions" in item
        assert len(item["versions"]) > 0
        if name == "ensemble":
            assert item["model_name"] == "ensemble"
            assert item["model_type"] == "ensemble"
            assert len(item["versions"]) == 2


@pytest.mark.order(after="test_create_models_from_zip_already_existing")
@pytest.mark.parametrize("name", [""])
def test_get_models_wrong_params(test_client, name):
    query_params = {}
    if name is not None:
        query_params["model_name"] = name

    request_uri = "/models"
    response = test_client.get(request_uri, params=query_params)
    assert response.status_code == 422


@pytest.mark.order(after="test_create_models_from_zip_already_existing")
@pytest.mark.parametrize("name", ["onnx", "python"])
def test_get_model(test_client, name):
    response = test_client.get(f"/models/{name}")

    assert response.status_code == 200
    data = response.json()
    assert data["model_name"] == name
    assert len(data["versions"]) > 0


@pytest.mark.order(after="test_get_models")
@pytest.mark.parametrize(
    "name, expected_code",
    [
        ("ensemble", 200),
        ("non_existent", 404),
    ],
)
def test_get_model_wrong_params(test_client, name, expected_code):
    response = test_client.get(f"/models/{name}")
    assert response.status_code == expected_code


@pytest.mark.order(after="test_get_models")
@pytest.mark.parametrize(
    "query_params",
    [
        {"model_name": "onnx"},
        {"model_name": "ensemble"},
    ],
)
def test_get_models_filters(test_client, query_params):
    response = test_client.get("/models", params=query_params)
    assert response.status_code == 200
    data = response.json()
    assert len(data) > 0
    for item in data:
        assert item["model_name"] == query_params["model_name"]


@pytest.mark.order(after="test_get_models_filters")
@pytest.mark.parametrize(
    "model_name, update_data, expected_status",
    [
        ("onnx", {"name": 12}, 422),
        ("onnx", {"name": ""}, 422),
        ("python", {"name": "updated_model"}, 200),
        ("updated_model", {"name": "python", "source": "whatever"}, 200),
        ("non_existent_model", {"name": "new_name"}, 404),
    ],
)
def test_rename_model(test_client, model_name, update_data, expected_status):
    response = test_client.put(f"/models/{model_name}", json=update_data)
    data = response.json()
    LOG.debug(f"Response: {data}")
    assert response.status_code == expected_status
    if expected_status == 200:
        assert "versions" in data
        for version in data["versions"]:
            path = version["model_uri"]
            assert update_data["name"] in path
            assert os.path.exists(path)


@pytest.mark.order(after="test_get_models_wrong_params")
@pytest.mark.parametrize("name", ["non_existent"])
def test_delete_non_existent_model(name, test_client):
    response = test_client.delete(f"/models/{name}")
    assert response.status_code == 404


@pytest.mark.order(after="test_get_models_wrong_params")
@pytest.mark.parametrize("name,version", [("python", 2)])
def test_delete_model_version(name, version, test_client, test_settings):
    params = {"model_version": version}
    response = test_client.delete(f"/models/{name}", params=params)
    assert response.status_code == 200
    data = response.json()
    assert data["deleted_at"] is None
    expected_path = test_settings.repository_path / name / str(version)
    assert not expected_path.exists()


@pytest.mark.order(after="test_delete_model_version")
@pytest.mark.parametrize("name", ["python"])
def test_delete_model_not_in_use(name, test_client, test_settings):
    response = test_client.delete(f"/models/{name}")
    assert response.status_code == 200
    data = response.json()
    assert data["deleted_at"] is not None
    expected_path = test_settings.repository_path / name
    assert not expected_path.exists()


def test_pyproject_supplies_pip_and_system_dependencies():
    deps = parse_dependencies(DATA / "bundle_pyproject")
    assert deps.pip == ["numpy==1.26.4", "pillow==10.0.0"]
    assert deps.system == ["libgl1", "libglib2.0-0"]


def test_unknown_tool_serve_keys_are_ignored():
    # `runtime` belongs to issue #130; a bundle carrying it must still parse here
    assert parse_dependencies(DATA / "bundle_pyproject").system == ["libgl1", "libglib2.0-0"]


def test_requirements_txt_is_still_read_when_no_pyproject():
    deps = parse_dependencies(DATA / "bundle_requirements")
    assert deps.pip == ["numpy==1.26.4"]
    assert deps.system == []


def test_pyproject_wins_over_requirements_txt(tmp_path: Path):
    (tmp_path / "requirements.txt").write_text("legacy==1.0.0\n")
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "b"\nversion = "0.1.0"\ndependencies = ["modern==2.0.0"]\n'
    )
    assert parse_dependencies(tmp_path).pip == ["modern==2.0.0"]


def test_missing_manifest_is_no_dependencies(tmp_path: Path):
    deps = parse_dependencies(tmp_path)
    assert deps.pip == []
    assert deps.system == []


def test_incompatible_requires_python_is_rejected():
    with pytest.raises(AssertionError, match="requires-python"):
        parse_dependencies(DATA / "bundle_bad_python")


def test_malformed_pyproject_is_rejected(tmp_path: Path):
    (tmp_path / "pyproject.toml").write_text("[project\nname =")
    with pytest.raises(AssertionError, match="pyproject.toml"):
        parse_dependencies(tmp_path)


def test_changed_dependencies_repoint_affected_services(test_client, test_db, test_settings):
    from triton_serve.api.models.domain import refresh_service_images
    from triton_serve.database.model import Model, Service

    response = test_client.post(
        "/services",
        json={
            "name": "trt-srv_test_fanout",
            "models": ["onnx"],
            "resources": {"gpus": 0, "shm_size": 256, "mem_size": 1024},
            "timeout": 3600,
        },
    )
    assert response.status_code == 201, response.text
    service_id = response.json()["service_id"]
    model = test_db.query(Model).filter(Model.model_name == "onnx").one()
    original = list(model.dependencies or [])
    try:
        test_db.expire_all()
        before = test_db.get(Service, ident=service_id).image_hash

        model.dependencies = ["numpy==1.26.0"]
        test_db.commit()
        refresh_service_images(db=test_db, models=[model], settings=test_settings)

        test_db.expire_all()
        service = test_db.get(Service, ident=service_id)
        assert service.image_hash != before
        assert service.image.managed
        assert "numpy==1.26.0" in service.image.pip_packages
    finally:
        model.dependencies = original
        test_db.commit()
        test_client.delete(f"/services/{service_id}")
