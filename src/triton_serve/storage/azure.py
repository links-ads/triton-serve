import logging
import time
from pathlib import Path
from shutil import rmtree

from azure.core.exceptions import ResourceNotFoundError  # pyright: ignore[reportMissingImports]
from azure.storage.blob import (  # pyright: ignore[reportMissingImports]
    BlobClient,
    BlobServiceClient,
    ContainerClient,
)

from triton_serve.storage.base import (
    ModelStorage,
    ModelStorageError,
    StorableModel,
    StorableVersion,
    StorageURI,
    WorkerRepository,
)

LOG = logging.getLogger("uvicorn")
CONFIG_FILE = "config.pbtxt"
COPY_POLL_INTERVAL = 0.5
COPY_POLL_TIMEOUT = 300


class AzureModelStorage(ModelStorage):
    """Storage backed by a flat Azure blob container.

    Directories do not exist here: a model is the set of blobs sharing a name prefix, and moving one
    is a server-side copy per blob followed by a delete. That is not atomic, which the design
    tolerates because the database is authoritative, copies are idempotent, soft delete retains
    deletions for seven days, and #149 sweeps what is left. Every step logs its prefix and blob
    count so partial state is diagnosable.
    """

    def __init__(
        self,
        account: str,
        container: str,
        credential: str,
        prefix: str,
        stash_prefix: str = ".stash",
        endpoint: str = "",
    ) -> None:
        self.account = account
        self.container = container
        self.prefix = prefix.strip("/")
        self.stash_prefix = stash_prefix.strip("/")
        self._check_prefixes()
        self._credential = credential
        self.client = BlobServiceClient(
            account_url=endpoint or f"https://{account}.blob.core.windows.net",
            credential={"account_name": account, "account_key": credential},
        )
        self._container: ContainerClient = self.client.get_container_client(container)

    def _check_prefixes(self) -> None:
        """Guards the invariant the stash rests on: it must sit outside what the workers can see.

        The settings layer enforces the same thing, but this class is also constructed directly, and
        a stash reachable from the repository prefix is a silent correctness bug rather than a
        startup failure.
        """
        if not self.prefix:
            raise ValueError("azure storage needs a non-empty prefix: the stash lives beside it")
        if not self.stash_prefix:
            raise ValueError("azure storage needs a non-empty stash prefix")
        if self.stash_prefix == self.prefix or self.stash_prefix.startswith(f"{self.prefix}/"):
            raise ValueError(f"the stash prefix {self.stash_prefix!r} sits inside the repository prefix")

    def _key(self, *parts: str) -> str:
        return "/".join(part for part in (self.prefix, *parts) if part)

    def _model_prefix(self, model_name: str) -> str:
        return self._key(model_name)

    def _stash_key(self, model_name: str) -> str:
        # outside self.prefix by construction: the workers are pointed at the prefix, never here
        return f"{self.stash_prefix}/{model_name}"

    def _blobs_under(self, prefix: str) -> list[str]:
        return [blob.name for blob in self._container.list_blobs(name_starts_with=f"{prefix}/")]

    def _move(self, source_prefix: str, destination_prefix: str) -> None:
        """Copies every blob under one prefix to another, then deletes the originals.

        Same-account copies authorise the source read with the request's own shared key, so no SAS
        is minted and no bytes pass through this process. Model weights routinely exceed the 256 MiB
        ceiling on a synchronous Copy Blob From URL, so the copy is started async and polled to
        completion; nothing is deleted until every copy in the batch is confirmed, so a failed or
        slow copy never costs the only copy of a model.
        """
        names = self._blobs_under(source_prefix)
        LOG.info("moving %d blobs from %s to %s", len(names), source_prefix, destination_prefix)
        destinations: list[BlobClient] = []
        for name in names:
            source = self._container.get_blob_client(name)
            destination = self._container.get_blob_client(f"{destination_prefix}{name[len(source_prefix) :]}")
            destination.start_copy_from_url(source.url)
            destinations.append(destination)
        for destination in destinations:
            self._await_copy(destination)
        for name in names:
            self._container.delete_blob(name)

    def _await_copy(self, blob: BlobClient) -> None:
        deadline = time.monotonic() + COPY_POLL_TIMEOUT
        while True:
            status = blob.get_blob_properties().copy.status
            if status == "success":
                return
            if status in ("failed", "aborted"):
                raise ModelStorageError(f"Copy to {blob.blob_name} {status}")
            if time.monotonic() >= deadline:
                raise ModelStorageError(f"Copy to {blob.blob_name} did not complete within {COPY_POLL_TIMEOUT}s")
            time.sleep(COPY_POLL_INTERVAL)

    def location(self, model: StorableModel, version: StorableVersion) -> StorageURI:
        return f"as://{self.account}/{self.container}/{self._key(model.model_name, str(version.version_id))}"

    def save(self, model: StorableModel, version: StorableVersion, origin: Path) -> StorageURI:
        version_tmp = origin / model.model_name / str(version.version_id)
        config_tmp = origin / model.model_name / CONFIG_FILE
        if not version_tmp.exists():
            raise ModelStorageError(f"Version {model.model_name}:{version.version_id} does not exist")
        config_key = self._key(model.model_name, CONFIG_FILE)
        # a bundle that omits the config only registers against a model that already has one
        if not config_tmp.exists() and not self._container.get_blob_client(config_key).exists():
            raise ModelStorageError(f"Missing config file in {model.model_name}")

        version_prefix = self._key(model.model_name, str(version.version_id))
        for file in sorted(path for path in version_tmp.rglob("*") if path.is_file()):
            key = f"{version_prefix}/{file.relative_to(version_tmp).as_posix()}"
            with file.open("rb") as handle:
                self._container.upload_blob(name=key, data=handle, overwrite=True)
        # stash moves the config away on update, so the partial-version window opens only at 2..N
        if config_tmp.exists():
            with config_tmp.open("rb") as handle:
                self._container.upload_blob(name=config_key, data=handle, overwrite=True)
            config_tmp.unlink()
        # consumed, mirroring the local backend's move: the origin is a one-shot temp bundle
        rmtree(version_tmp)
        return self.location(model, version)

    def delete(self, model: StorableModel, version: StorableVersion) -> None:
        version_prefix = self._key(model.model_name, str(version.version_id))
        names = self._blobs_under(version_prefix)
        # the local backend rmtree's a missing directory; the contract says both raise
        if not names:
            raise FileNotFoundError(f"Version {model.model_name}:{version.version_id} has no blobs")
        for name in names:
            self._container.delete_blob(name)
        model_prefix = self._model_prefix(model.model_name)
        config_key = self._key(model.model_name, CONFIG_FILE)
        remaining = [n for n in self._blobs_under(model_prefix) if n != config_key]
        if not remaining:
            for name in self._blobs_under(model_prefix):
                self._container.delete_blob(name)

    def rename(self, model: StorableModel, new_name: str) -> None:
        source = self._model_prefix(model.model_name)
        destination = self._model_prefix(new_name)
        if not self._blobs_under(source):
            raise FileNotFoundError(f"Model '{model.model_name}' has no blobs under {source}")
        if self._blobs_under(destination):
            raise FileExistsError(f"Model '{new_name}' already has blobs under {destination}")
        self._move(source, destination)

    def stash(self, model: StorableModel) -> StorageURI:
        source = self._model_prefix(model.model_name)
        destination = self._stash_key(model.model_name)
        if not self._blobs_under(source):
            raise FileNotFoundError(f"Model '{model.model_name}' has no blobs under {source}")
        if self._blobs_under(destination):
            raise FileExistsError(f"A stash is already outstanding for '{model.model_name}' at {destination}")
        self._move(source, destination)
        return f"as://{self.account}/{self.container}/{destination}"

    def restore(self, model: StorableModel, stashed: StorageURI) -> None:
        source = self._relative(stashed)
        if not self._blobs_under(source):
            raise FileNotFoundError(f"No stash at {stashed}")
        destination = self._model_prefix(model.model_name)
        for name in self._blobs_under(destination):
            self._container.delete_blob(name)
        self._move(source, destination)

    def discard(self, stashed: StorageURI) -> None:
        try:
            names = self._blobs_under(self._relative(stashed))
        except ResourceNotFoundError:
            return
        for name in names:
            try:
                self._container.delete_blob(name)
            except ResourceNotFoundError:
                continue

    def exists(self, uri: StorageURI) -> bool:
        relative = self._relative(uri)
        if self._blobs_under(relative):
            return True
        return self._container.get_blob_client(relative).exists()

    def read(self, uri: StorageURI) -> bytes:
        relative = self._relative(uri)
        try:
            return self._container.download_blob(relative).readall()
        except ResourceNotFoundError as error:
            raise FileNotFoundError(f"no blob at {relative!r} in container {self.container!r}") from error

    def check_reachable(self) -> None:
        try:
            self._container.get_container_properties()
        except Exception as error:
            raise ConnectionError(
                f"azure container {self.container!r} on account {self.account!r} is unreachable "
                f"(prefix {self.prefix!r}): {error}"
            ) from error

    def worker_repository(self) -> WorkerRepository:
        uri = f"as://{self.account}/{self.container}/{self.prefix}".rstrip("/")
        return WorkerRepository(
            uri=uri,
            mounts={},
            environment={
                "WORKER_REPOSITORY": uri,
                "AZURE_STORAGE_ACCOUNT": self.account,
                "AZURE_STORAGE_KEY": self._credential,
            },
        )

    def _relative(self, uri: StorageURI) -> str:
        return uri.removeprefix(f"as://{self.account}/{self.container}/")
