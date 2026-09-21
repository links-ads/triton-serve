import hashlib
import io
import logging
import os
import re
from collections.abc import Callable
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import TYPE_CHECKING
from zipfile import ZipFile

import docker
import pytest
import python_multipart
import urllib3
from fastapi import UploadFile
from httpx import Client

from triton_serve.builder.spec import BuildSpec, make_build_spec
from triton_serve.config import get_settings
from triton_serve.database import database_manager
from triton_serve.storage.local import LocalModelStorage
from triton_serve.storage.sources import ArchiveModelSource

if TYPE_CHECKING:
    from triton_serve.storage.azure import AzureModelStorage

logging.getLogger(python_multipart.__name__).setLevel(logging.WARNING)
logging.getLogger(docker.__name__).setLevel(logging.WARNING)
logging.getLogger(urllib3.__name__).setLevel(logging.WARNING)
LOG = logging.getLogger(pytest.__name__)

TEST_DIR = Path(os.getenv("TEST_DIR", Path(__file__).parent))
TEST_GIT_REPO = os.getenv("TEST_GIT_REPO")
ARCHIVE_NAME = "repository.zip"
BASE_IMAGE = "ghcr.io/links-ads/serve-triton:23.07-py3"


@pytest.fixture
def build_spec() -> Callable[..., BuildSpec]:
    """Builds a spec on shared defaults, so a test only spells out what it is actually about."""

    def _spec(**kwargs) -> BuildSpec:
        return make_build_spec(**{"base_image": BASE_IMAGE, "apt_packages": [], "pip_packages": [], **kwargs})

    return _spec


AZURITE = os.getenv("AZURITE_ENDPOINT", "")


def _sanitize_container_name(name: str, prefix: str = "") -> str:
    """Maps a pytest node id to a legal, unique Azure container name.

    Truncating the collapsed name alone can collide (two parameterised cases sharing a long
    common prefix), so a short digest of the full name is appended to keep every case distinct.
    """
    collapsed = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    digest = hashlib.blake2s(name.encode(), digest_size=3).hexdigest()
    budget = 63 - len(prefix) - len(digest) - 1
    body = collapsed[:budget].strip("-")
    return f"{prefix}{body}-{digest}"


def _azure_storage(request: pytest.FixtureRequest) -> AzureModelStorage:
    from azure.core.exceptions import ResourceExistsError
    from azure.storage.blob import BlobServiceClient

    from triton_serve.storage.azure import AzureModelStorage

    # one container per test, so the session-scoped suite cannot leak state between cases
    container = _sanitize_container_name(request.node.name, prefix="conf-")
    storage = AzureModelStorage(
        account=os.environ["AZURITE_ACCOUNT"],
        container=container,
        credential=os.environ["AZURITE_KEY"],
        endpoint=AZURITE,
    )
    client: BlobServiceClient = storage.client
    # self-healing: a previous run that died before its finalizer ran would otherwise poison this one
    with suppress(ResourceExistsError):
        client.create_container(container)
    request.addfinalizer(lambda: client.delete_container(container))
    return storage


@pytest.fixture(params=["local", "azure"])
def storage(request: pytest.FixtureRequest, tmp_path: Path) -> LocalModelStorage | AzureModelStorage:
    if request.param == "local":
        repository = tmp_path / "models"
        repository.mkdir()
        return LocalModelStorage(repository)
    if not AZURITE:
        pytest.skip("AZURITE_ENDPOINT is not set; Azurite is only up under `make test`")
    return _azure_storage(request)


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
        containers = [c for c in client.containers.list() if c.name.startswith("trt-srv_test_")]  # type: ignore
        for container in containers:
            LOG.info(f"Removing container {container.name}...")
            container.stop()
            container.remove()
            LOG.info(f"Container {container.name} removed.")
    finally:
        client.close()


@pytest.fixture
def bundle_source() -> Callable[..., ArchiveModelSource]:
    """Builds a registrable bundle in memory, so a test spells out only what it is about."""

    def _source(models: dict[str, bytes], archive_name: str = ARCHIVE_NAME) -> ArchiveModelSource:
        archive = io.BytesIO()
        with ZipFile(archive, "w") as f:
            for name, payload in models.items():
                f.writestr(f"model_repository/{name}/config.pbtxt", "")
                f.writestr(f"model_repository/{name}/1/model.onnx", payload)
            f.writestr(
                "pyproject.toml",
                '[project]\nname = "test-bundle"\nversion = "0.1.0"\nrequires-python = ">=3.10"\ndependencies = []\n',
            )
        archive.seek(0)
        upload = UploadFile(file=archive, filename=archive_name)
        return ArchiveModelSource(upload, target_dir="model_repository")

    return _source


@pytest.fixture(scope="session")
def make_zip() -> Callable:
    @contextmanager
    def _create_zip(
        archive_name: str = ARCHIVE_NAME,
        include_models: list[str] | None = None,
        exclude_models: list[str] | None = None,
        include_files: list[str] | None = None,
        exclude_files: list[str] | None = None,
        include_manifest: bool = True,
    ):
        """Utility function to create a zip file with the given models/files.

        Args:
            archive_name (str): name of the archive
            include_models (list[str], optional): list of models to include. Defaults to None.
            exclude_models (list[str], optional): list of models to exclude. Defaults to None.
            include_files (list[str], optional): list of files to include. Defaults to None.
            exclude_files (list[str], optional): list of files to exclude. Defaults to None.
            include_manifest (bool, optional): whether to add a root pyproject.toml. Defaults to True.
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
                if include_manifest:
                    f.writestr(
                        "pyproject.toml",
                        '[project]\nname = "test-bundle"\nversion = "0.1.0"\n'
                        'requires-python = ">=3.10"\ndependencies = []\n',
                    )
            archive.seek(0)
            yield archive
        finally:
            if archive:
                archive.close()

    return _create_zip
