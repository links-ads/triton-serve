import io
import logging
import os
from contextlib import contextmanager
from pathlib import Path
from zipfile import ZipFile

import docker
import multipart
import pytest
import urllib3
from fastapi.testclient import TestClient

from src.triton_serve.config import get_settings
from src.triton_serve.factory import create_app

logging.getLogger(multipart.__name__).setLevel(logging.WARNING)
logging.getLogger(docker.__name__).setLevel(logging.WARNING)
logging.getLogger(urllib3.__name__).setLevel(logging.WARNING)
LOG = logging.getLogger(pytest.__name__)

TEST_DIR = os.getenv("TEST_DIR", Path(__file__).parent)


@pytest.fixture(scope="session")
def test_settings():
    settings = get_settings()
    yield settings


@pytest.fixture(scope="session")
def test_dir():
    return TEST_DIR


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


@pytest.fixture(scope="session")
def test_docker():
    client = docker.from_env()
    try:
        yield client
        containers = [c for c in client.containers.list() if c.name.startswith("trt-srv_test_sv")]
        for container in containers:
            LOG.info(f"Removing container {container.name}...")
            container.stop()
            container.remove()
            LOG.info(f"Container {container.name} removed.")
    finally:
        client.close()


@pytest.fixture(scope="session")
def make_zip():
    @contextmanager
    def _create_zip(
        model_name: str = "model.onnx",
        add_model: bool = True,
        add_config: bool = False,
    ):
        """Utility function to create a zip file with the given files.

        Args:
            archive_name (str): name of the archive
            config_file (bool, optional): whether to include a config file. Defaults to False.
        """
        archive = io.BytesIO()
        archive.name = "package.zip"
        data_dir = TEST_DIR / "data"
        try:
            with ZipFile(archive, "w") as f:
                if add_model:
                    f.write(data_dir / "model.onnx", arcname=Path(model_name))
                if add_config:
                    f.write(data_dir / "config.pbtxt", arcname=Path("config.pbtxt"))
            archive.seek(0)
            yield archive
        finally:
            if archive:
                archive.close()

    return _create_zip
