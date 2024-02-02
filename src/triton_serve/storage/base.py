from abc import ABC, abstractmethod
from pathlib import Path

from fastapi import UploadFile

from triton_serve.database.schema import ModelSchema


class ModelStorage(ABC):
    supported_formats = [".zip", ".tar", ".gz"]

    def __init__(self, base_path: Path) -> None:
        self.base_path = base_path

    def location(self, model: ModelSchema) -> str:
        """Constructs the absolute path to the package, starting from the base path
        and using both the artifact's name and the artifact's version.

        :param artifact: artifact instance or its schema
        :type artifact: ArtifactSchema
        :return: path to the artifact files
        :rtype: str
        """
        return self.base_path / model.model_name / str(model.model_version)

    @abstractmethod
    def check_format(self, filename: str) -> None:
        """Checks whether the given filename respects the expected format.

        Args:
            filename (str): file name
        """
        ...

    @abstractmethod
    def load(self, model: ModelSchema) -> Path:
        """Required to transform a possibly remote URI into a local path.
        No-op for local storage.

        Args:
            model (ModelSchema): model name and version

        Returns:
            Path: local path to a model.
        """
        ...

    @abstractmethod
    def exists(self, model: ModelSchema) -> bool:
        """Checks whether the given model exists.

        Args:
            model (ModelSchema): model name and version

        Returns:
            bool: True if the model exists, False otherwise.
        """
        ...

    @abstractmethod
    def save(self, model: ModelSchema, package: UploadFile) -> Path:
        """Required to store the UploadFile into the storage implementation (locally, blog storage, etc.).
        This is the complement of the load method.

        Args:
            model (ModelSchema): model name and version.
            package (UploadFile): file or package to be uploaded.

        Returns:
            Path: local or remote path to the model.
        """
        ...

    @abstractmethod
    def update(self, previous: ModelSchema, updated: ModelSchema) -> Path:
        """Required to update a given URI and move files around.
        Generates a new URI for the updated model.

        Args:
            previous (ModelSchema): previous model name and version.
            updated (ModelSchema): current model name and version.

        Returns:
            Path: local or remote path to the model.
        """
        raise NotImplementedError()

    @abstractmethod
    def delete(self, model: ModelSchema) -> None:
        """Deletes the given model.

        Args:
            model (ModelSchema): model name and version.
        """
        ...

    @abstractmethod
    def close(self) -> None:
        """Closes the storage instance."""
        ...
