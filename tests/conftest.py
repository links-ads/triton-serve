import logging
import zipfile
from pathlib import Path

import coloredlogs
import pytest
from fastapi.testclient import TestClient

from src.triton_serve.config import get_settings
from src.triton_serve.config.schema import AppSettings
from src.triton_serve.extensions import docker_client
from src.triton_serve.factory import create_app
from src.triton_serve.storage.local import LocalModelStorage

LOG = logging.getLogger(pytest.__name__)
coloredlogs.install(logger=LOG, isatty=True)


@pytest.fixture(scope="session")
def test_settings():
    settings = get_settings()
    yield settings


@pytest.fixture(scope="session")
def test_storage(test_settings):
    storage = LocalModelStorage(test_settings.repository_path)
    yield storage


@pytest.fixture(scope="session")
def test_app():
    LOG.info("Initializing webserver...")
    app = create_app(settings=get_settings())
    yield app


@pytest.fixture(scope="session")
def test_client(test_app):
    LOG.debug("Initializing test client...")
    client = TestClient(app=test_app)
    yield client


def _create_zip(
    test_settings: AppSettings,
    archive_name: str,
    model_filename: str = "model.onnx",
    model_file: bool = True,
    config_file: bool = False,
):
    """Utility function to create a zip file with the given files.

    Args:
        archive_name (str): name of the archive
        config_file (bool, optional): whether to include a config file. Defaults to False.
    """
    model_test_dir = test_settings.tests_dir / "data" / "model_test"
    zip_file = test_settings.zips_path / archive_name

    with zipfile.ZipFile(zip_file, "w") as model_zip:
        # write the files in the zip file
        if model_file:
            model_zip.write(model_test_dir / "model.onnx", arcname=Path(model_filename))
        if config_file:
            model_zip.write(model_test_dir / "config.pbtxt", arcname=Path("config.pbtxt"))


# Create a zip file with model.onnx and config.pbtxt
# In the tests/data folder, put a folder called model_test with the model.onnx and config.pbtxt files
@pytest.fixture(scope="session")
def create_zips(test_settings):
    _create_zip(test_settings, "onnx_and_config.zip", config_file=True)
    _create_zip(test_settings, "only_onnx.zip", config_file=False)
    _create_zip(test_settings, "wrong_model_name.zip", config_file=False, model_filename="wrong_model_name.onnx")
    _create_zip(test_settings, "no_model.zip", model_file=False, config_file=True)
    yield


# Instantiates the docker client and cleanes up the containers after the tests
@pytest.fixture(scope="session")
def test_containers():
    client = docker_client()
    spawned_containers_names = []
    yield {
        "client": client,
        "spawned_containers_names": spawned_containers_names,
    }
    # clean up containers
    for container_name in spawned_containers_names:
        LOG.debug(f"Removing container {container_name}...")
        container = client.containers.get(container_name)
        container.stop()
        container.remove()
        LOG.debug(f"Container {container_name} removed.")
