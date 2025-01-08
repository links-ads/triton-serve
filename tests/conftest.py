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
from httpx import Client
from triton_serve.config import get_settings
from triton_serve.database import database_manager

logging.getLogger(multipart.__name__).setLevel(logging.WARNING)
logging.getLogger(docker.__name__).setLevel(logging.WARNING)
logging.getLogger(urllib3.__name__).setLevel(logging.WARNING)
LOG = logging.getLogger(pytest.__name__)

TEST_DIR = os.getenv("TEST_DIR", Path(__file__).parent)
TEST_GIT_REPO = os.getenv("TEST_GIT_REPO")
ARCHIVE_NAME = "repository.zip"


@pytest.fixture(scope="session")
def test_repository():
    """
    Test if the repository is set, then yield the repository url
    """
    if not TEST_GIT_REPO:
        pytest.skip("No test repository provided")
    yield TEST_GIT_REPO


@pytest.fixture(scope="session")
def test_archive():
    """
    Test if the repository is set, then yield the repository url
    """
    if not ARCHIVE_NAME:
        pytest.skip("No test repository provided")
    yield ARCHIVE_NAME


@pytest.fixture(scope="session")
def test_settings():
    """
    Get the settings
    """
    settings = get_settings()
    yield settings


@pytest.fixture(scope="session")
def test_connection(test_settings):
    """
    Get the database connection

    :return: the database connection

    """
    database_manager.init(test_settings.database_url)
    yield
    database_manager.close()


@pytest.fixture(scope="session")
def test_db(test_connection):
    """
    Get the database session

    :return: the database session
    """
    with database_manager.session() as session:
        yield session


@pytest.fixture(scope="session")
def test_client(test_settings, custom_headers=None, timeout=60):
    LOG.debug("Initializing test client...")
    client = Client(base_url=f"http://{test_settings.backend_host}:{test_settings.backend_port}", timeout=timeout)
    client.headers.update({"X-API-Key": test_settings.api_keys[0]})
    if custom_headers:
        client.headers.update(custom_headers)
    yield client
    client.close()


@pytest.fixture(scope="session")
def test_docker():
    """
    Get the docker client to create and delete containers
    """
    client = docker.from_env()
    try:
        yield client
        containers = [c for c in client.containers.list() if c.name.startswith("trt-srv_test_")]
        for container in containers:
            LOG.info(f"Removing container {container.name}...")
            container.stop()
            container.remove()
            LOG.info(f"Container {container.name} removed.")
    finally:
        client.close()


@pytest.fixture(scope="session")
def make_zip() -> io.BytesIO:
    @contextmanager
    def _create_zip(
        archive_name: str = ARCHIVE_NAME,
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
