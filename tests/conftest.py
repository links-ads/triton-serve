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

from triton_serve.extensions import get_db

logging.getLogger(multipart.__name__).setLevel(logging.WARNING)
logging.getLogger(docker.__name__).setLevel(logging.WARNING)
logging.getLogger(urllib3.__name__).setLevel(logging.WARNING)
LOG = logging.getLogger(pytest.__name__)

TEST_DIR = os.getenv("TEST_DIR", Path(__file__).parent)
TEST_GIT_REPO = os.getenv("TEST_GIT_REPO")


@pytest.fixture(scope="session")
def test_repository():
    """
    Test if the repository is set, then yield the repository url
    """
    if not TEST_GIT_REPO:
        pytest.skip("No test repository provided")
    yield TEST_GIT_REPO


@pytest.fixture(scope="function")
def test_db():
    """
    Get the database connection

    :return: the database connection

    """
    db = next(get_db())
    yield db


@pytest.fixture(scope="session")
def test_settings():
    """
    Get the settings
    """
    settings = get_settings()
    yield settings


@pytest.fixture(scope="session")
def test_app():
    """
    Get the uvicorn test app

    :return: the test app
    """
    LOG.info("Initializing webserver...")
    app = create_app(settings=get_settings())
    yield app


@pytest.fixture(scope="session")
def test_client(test_app, test_settings):
    """
    Get the test client to call the endpoints of the webserver
    """
    LOG.debug("Initializing test client...")
    client = TestClient(app=test_app)
    client.headers.update({"X-API-Key": test_settings.app_secret})
    yield client


@pytest.fixture(scope="session")
def test_docker():
    """
    Get the docker client to create and delete containers
    """
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
        archive_name: str = "repository.zip",
        include_models: list[str] = None,
        exclude_models: list[str] = None,
        include_files: list[str] = None,
        exclude_files: list[str] = None,
    ):
        """Utility function to create a zip file with the given models/files.

        Args:
            archive_name (str): name of the archive
            include_models (list[str], optional): list of models to include. Defaults to None.
            exclude_models (list[str], optional): list of models to exclude. Defaults to None.
            include_files (list[str], optional): list of files to include. Defaults to None.
            exclude_files (list[str], optional): list of files to exclude. Defaults to None.
        """
        archive = io.BytesIO()
        archive.name = archive_name
        data_dir = TEST_DIR / "data"
        repository_dir = data_dir / "model_repository"
        model_dirs = [d for d in repository_dir.iterdir() if d.is_dir()]

        try:
            with ZipFile(archive, "w") as f:
                if include_models is not None:
                    model_dirs = [d for d in model_dirs if d.name in include_models]
                if exclude_models is not None:
                    model_dirs = [d for d in model_dirs if d.name not in exclude_models]
                # on each model, recursively gather each file or directory
                for model_dir in model_dirs:
                    model_files = list(model_dir.rglob("*"))
                    if include_files is not None:
                        model_files = [f for f in model_files if f.name in include_files]
                    if exclude_files is not None:
                        model_files = [f for f in model_files if f.name not in exclude_files]
                    for model_file in model_files:
                        f.write(model_file, arcname=model_file.relative_to(data_dir))
            archive.seek(0)
            yield archive
        finally:
            if archive:
                archive.close()

    return _create_zip
