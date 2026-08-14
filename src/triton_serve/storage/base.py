from abc import ABC, abstractmethod
from collections.abc import Iterator
from pathlib import Path
from tarfile import TarFile
from zipfile import ZipFile

from triton_serve.database.schema import ModelSchema, ModelVersionSchema


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


class ModelSource(ABC):
    """Generic class to represent a source of models."""

    @abstractmethod
    def origin(self) -> str:
        """Returns the origin of the models (filename, URL, etc.)

        Returns:
            str: origin of the models.
        """
        ...

    @abstractmethod
    def extract(self, path: Path) -> Path:
        """Extracts the models from the source.

        Returns:
            Path: local path to the folder with the extracted models.
        """
        ...


class ModelStorage(ABC):
    """Where model files live. Placing them is the whole job.

    Triton reads the model repository straight off a shared volume, so nothing here ever hands
    model bytes to a caller -- a storage backend only has to put files where Triton will find them.
    """

    def __init__(self, base_path: Path) -> None:
        self.base_path: Path = base_path

    def location(self, model: ModelSchema, version: ModelVersionSchema) -> Path:
        """Constructs the absolute path to the package, starting from the base path
        and using both the artifact's name and the artifact's version.

        Args:
            model (ModelSchema): the model to locate
            version (ModelVersionSchema): model version

        Returns:
            Path: absolute path to the package
        """
        return self.base_path / model.model_name / str(version.version_id)

    @abstractmethod
    def save(self, model: ModelSchema, version: ModelVersionSchema, origin: Path) -> Path:
        """Required to store the given data into the storage implementation (locally, blob storage, etc.).

        Args:
            model (ModelSchema): the model being stored.
            version (ModelVersionSchema): model version.
            origin (Path): local path to the model root.

        Returns:
            Path: path to the model root.
        """
        ...

    @abstractmethod
    def update(self, model: ModelSchema, version: ModelVersionSchema, current_uri: Path) -> Path:
        """Required to update a given URI and move files around.
        Generates a new URI for the updated model.

        Args:
            model (ModelSchema): current model name and version.
            version (ModelVersionSchema): current model version.
            current_uri (Path): old path to the model root, to be updated.

        Returns:
            Path: updated local or remote path to the model.
        """
        ...

    @abstractmethod
    def delete(self, model: ModelSchema, version: ModelVersionSchema) -> None:
        """Deletes the given model.

        Args:
            model (ModelSchema): model name and version.
            version (ModelVersionSchema): model version.
        """
        ...
