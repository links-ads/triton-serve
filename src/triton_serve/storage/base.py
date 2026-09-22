from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from tarfile import TarFile
from typing import Protocol
from zipfile import ZipFile

from triton_serve.storage.validation import BundleDependencies

type StorageURI = str


class ModelStorageError(Exception):
    """A bundle does not hold what it declared, so a backend cannot store it."""


class StorableModel(Protocol):
    """All a backend needs of a model: the name its files are filed under."""

    model_name: str


class StorableVersion(Protocol):
    """All a backend needs of a version: the number its files are filed under."""

    version_id: int


class BaseExtractor[ArchiveT: (ZipFile, TarFile)](ABC):
    """Base class for extracting files from an archive.

    Subclasses only supply the two things the archive libraries spell differently: how the archive
    is opened, and how its members are listed. Closing and extracting are identical for both.
    """

    archive: ArchiveT

    def __init__(self, file: Path):
        self.file = file

    @abstractmethod
    def _open(self) -> ArchiveT:
        """Opens the archive for reading."""
        ...

    @abstractmethod
    def __iter__(self) -> Iterator[str]:
        """Yields the names of the archive members."""
        ...

    def __enter__(self) -> BaseExtractor[ArchiveT]:
        self.archive = self._open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.archive.close()

    def extract(self, path: Path, member: str | None = None) -> None:
        """Extracts `member` into `path`, or the whole archive when no member is given."""
        if member is not None:
            self.archive.extract(member, path)
        else:
            self.archive.extractall(path)


@dataclass(frozen=True)
class ExtractedBundle:
    """What a source hands back: where the models are, and what the bundle declared."""

    models: Path
    dependencies: BundleDependencies


@dataclass(frozen=True)
class WorkerRepository:
    """How a worker container reaches the model repository.

    The storage backend owns this because only it knows whether the repository is a volume to mount
    or a URI to read over the network.
    """

    uri: str
    mounts: dict[str, dict[str, str]]
    environment: dict[str, str] = field(repr=False)


class ModelSource(ABC):
    """Generic class to represent a source of models."""

    def __init__(self, target_dir: str):
        self.target_dir = target_dir

    @abstractmethod
    def origin(self) -> str:
        """Returns the origin of the models (filename, URL, etc.)

        Returns:
            str: origin of the models.
        """
        ...

    @abstractmethod
    def extract(self, path: Path) -> ExtractedBundle:
        """Extracts the bundle from the source.

        Returns:
            ExtractedBundle: The models directory and the bundle's declared dependencies.
        """
        ...


class ModelStorage(ABC):
    """Where a model's files live, and how they are moved aside when a new version lands.

    Nothing here hands model bytes back to a caller: a backend only has to put files where the
    workers will find them, which may be a shared volume or an object store. Locations are
    therefore URIs, not paths.

    Every implementation raises the same exception for the same condition, so no call site needs to
    know which backend it holds: `ModelStorageError` for a bundle missing the version it declared,
    `FileNotFoundError` for an absent source, `FileExistsError` for an occupied destination.
    """

    @abstractmethod
    def location(self, model: StorableModel, version: StorableVersion) -> StorageURI:
        """Returns the URI a model version's files live at.

        Args:
            model (StorableModel): the model to locate.
            version (StorableVersion): the version to locate.

        Returns:
            StorageURI: where that version's files are, whether or not they exist yet.
        """
        ...

    @abstractmethod
    def save(self, model: StorableModel, version: StorableVersion, origin: Path) -> StorageURI:
        """Stores one version of a model, taking its files from a locally extracted bundle.

        Args:
            model (StorableModel): the model being stored.
            version (StorableVersion): the version being stored.
            origin (Path): local path to the extracted bundle's repository directory. The version
                subtree it stores, and the model's config, are consumed by this call; the origin may
                be reused for other versions of the same bundle.

        Returns:
            StorageURI: where the version now lives.

        Raises:
            ModelStorageError: if the bundle holds no such version, or no config for the model.
        """
        ...

    @abstractmethod
    def delete(self, model: StorableModel, version: StorableVersion) -> None:
        """Deletes one version, and the model itself once no version is left.

        Args:
            model (StorableModel): the model to delete from.
            version (StorableVersion): the version to delete.

        Raises:
            FileNotFoundError: if the version has no files.
        """
        ...

    @abstractmethod
    def rename(self, model: StorableModel, new_name: str) -> None:
        """Moves every file of a model from its current name to `new_name`.

        `model` still carries the old name when this is called; the caller renames the record only
        once this returns.

        Args:
            model (StorableModel): the model to move, under its current name.
            new_name (str): the name to move it to.

        Raises:
            FileNotFoundError: if the model has no files.
            FileExistsError: if `new_name` is already occupied.
        """
        ...

    @abstractmethod
    def stash(self, model: StorableModel) -> StorageURI:
        """Moves a model's files aside, out of the way of replacements about to be written.

        The stash lives under a reserved location the workers never load as a model.

        Args:
            model (StorableModel): the model whose files are moved aside.

        Returns:
            StorageURI: a handle to pass to `restore` or `discard`.

        Raises:
            FileNotFoundError: if the model has no files.
            FileExistsError: if a stash is already outstanding for this model, which means a
                previous transaction died and those files may be the only copy.
        """
        ...

    @abstractmethod
    def restore(self, model: StorableModel, stashed: StorageURI) -> None:
        """Puts a stash back, replacing anything a failed attempt left at the model's location.

        Args:
            model (StorableModel): the model to restore.
            stashed (StorageURI): the handle `stash` returned.

        Raises:
            FileNotFoundError: if the stash is gone.
        """
        ...

    @abstractmethod
    def discard(self, stashed: StorageURI) -> None:
        """Drops a stash whose transaction committed. Quiet if it is already gone.

        Args:
            stashed (StorageURI): the handle `stash` returned.
        """
        ...

    @abstractmethod
    def exists(self, uri: StorageURI) -> bool:
        """Reports whether anything is stored at a URI, whether a single file or a subtree.

        Args:
            uri (StorageURI): a URI this backend produced.

        Returns:
            bool: True if the URI resolves to stored content.
        """
        ...

    @abstractmethod
    def read(self, uri: StorageURI) -> bytes:
        """Reads back a single stored file.

        Args:
            uri (StorageURI): a URI this backend produced, naming one file.

        Returns:
            bytes: the file's contents.
        """
        ...

    @abstractmethod
    def worker_repository(self) -> WorkerRepository:
        """Returns the mounts and environment a worker needs to read this repository.

        Returns:
            WorkerRepository: the repository URI, the volumes to mount, and the environment that
                points the worker at it.
        """
        ...
