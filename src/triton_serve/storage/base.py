from abc import ABC, abstractmethod
from pathlib import Path

from triton_serve.database.schema import ModelSchema, ModelVersionSchema


class BaseExtractor(ABC):
    """Base class for extracting files from an archive."""

    def __init__(self, file: Path):
        self.file = file

    @abstractmethod
    def __enter__(self, mode: str = "r"): ...

    @abstractmethod
    def __exit__(self, exc_type, exc_val, exc_tb): ...

    @abstractmethod
    def __iter__(self): ...

    @abstractmethod
    def extract(self, member: str, path: Path): ...


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
    def extract(self) -> Path:
        """Extracts the models from the source.

        Returns:
            Path: local path to the folder with the extracted models.
        """
        ...


class ModelStorage(ABC):
    supported_formats = [".zip", ".tar", ".gz"]

    def __init__(self, base_path: Path) -> None:
        self.base_path = base_path

    def location(self, model: ModelSchema, version: ModelVersionSchema) -> str:
        """Constructs the absolute path to the package, starting from the base path
        and using both the artifact's name and the artifact's version.

        Args:
            model (ModelSchema): model name and version
            version (ModelVersionSchema): model version

        Returns:
            str: absolute path to the package
        """
        return self.base_path / model.model_name / str(version.version_id)

    @abstractmethod
    def load(self, model: ModelSchema, version: ModelVersionSchema) -> Path:
        """Required to transform a possibly remote URI into a local path.
        No-op for local storage.

        Args:
            model (ModelSchema): model name and version
            version (ModelVersionSchema): model version

        Returns:
            Path: local path to a model.
        """
        ...

    @abstractmethod
    def exists(self, model: ModelSchema, version: ModelVersionSchema) -> bool:
        """Checks whether the given model exists.

        Args:
            model (ModelSchema): model name and version
            version (ModelVersionSchema): model version

        Returns:
            bool: True if the model exists, False otherwise.
        """
        ...

    @abstractmethod
    def save(self, model: ModelSchema, version: ModelVersionSchema, origin: Path) -> Path:
        """Required to store the given data into the storage implementation (locally, blog storage, etc.).
        This is the complement of the load method.

        Args:
            model (ModelSchema): model name and version.
            verrsion (ModelVersionSchema): model version.
            origin (Path): local path to the model root.

        Returns:
            path to the model root.
        """
        ...

    @abstractmethod
    def update(self, model: ModelSchema, version: ModelVersionSchema, current_uri: Path) -> Path:
        """Required to update a given URI and move files around.
        Generates a new URI for the updated model.

        Args:
            model (ModelSchema): current model name and version.
            version (ModelVersionSchema): current model version.
            origin (Path): old path to the model root, to be updated.

        Returns:
            Path: updated local or remote path to the model.
        """
        raise NotImplementedError()

    @abstractmethod
    def delete(self, model: ModelSchema, version: ModelVersionSchema) -> None:
        """Deletes the given model.

        Args:
            model (ModelSchema): model name and version.
            version (ModelVersionSchema): model version.
        """
        ...

    @abstractmethod
    def close(self) -> None:
        """Closes the storage instance."""
        ...
