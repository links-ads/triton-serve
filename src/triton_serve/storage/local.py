import tempfile
from collections.abc import Iterator
from pathlib import Path
from shutil import copyfileobj, move, rmtree
from tarfile import TarFile
from zipfile import ZipFile

from fastapi import UploadFile

from triton_serve.database.schema import ModelSchema
from triton_serve.storage import ModelStorage


class LocalModelStorage(ModelStorage):
    """Storage implementation based on a local file system.
    This module handles a single path, managing file upload and extraction, update and deletion.
    """

    def __init__(self, base_path: str) -> None:
        """Creates a new storage module instance, instantiating attributes.

        Args:
            base_path (str): Path to the root model folder.
        """
        super().__init__(base_path=base_path)
        self.extractor = None
        self.read_mode = None

    def check_format(self, filename: Path) -> None:
        """Extracts the extension from the filename, assigning the correct extraction
        and read modes to the instance.

        Args:
            filename (str): Name of the uploaded file.

        Raises:
            ValueError: When the file extension is not among the supported ones.
        """
        if filename.endswith(".zip"):
            self.extractor = ZipFile
            self.read_mode = "r"
        elif filename.endswith(".tar.gz") or filename.endswith(".tgz"):
            self.extractor = TarFile
            self.read_mode = "r:gz"
        else:
            suffix = Path(filename).suffix
            raise ValueError(f"Unsupported file format: {suffix}")

    def _iterate(self, file: TarFile | ZipFile) -> Iterator[str]:
        """Generator function that abstracts from the archive type and provides iteration
        functionalities for both zip and tar archives.

        Args:
            file (Union[TarFile, ZipFile]): Archive to be iterated.

        Yields:
            str: String corresponding to the item names.
        """
        if isinstance(file, TarFile):
            for item in file.getmembers():
                yield Path(item.name)
        elif isinstance(file, ZipFile):
            for item in file.namelist():
                yield Path(item)

    def _delete_empty_directories(self, base_path: Path) -> None:
        """
        Deletes empty directories within the given path.

        Args:
            base_path (Path): The base path to start searching for empty directories.
        """
        for dir_path in base_path.iterdir():
            if dir_path.is_dir():
                self._delete_empty_directories(dir_path)
                if not any(dir_path.iterdir()):
                    dir_path.rmdir()

    def load(self, model: ModelSchema) -> str:
        return self.location(model)

    def exists(self, model: ModelSchema) -> bool:
        return Path(self.location(model)).exists()

    def save(self, model: ModelSchema, package: UploadFile) -> str:
        self.check_format(filename=package.filename)
        model_path = self.location(model)

        with tempfile.TemporaryDirectory() as temp_folder:
            tmp_file = Path(temp_folder) / package.filename
            # logger.debug("Storing %s in ...", tmp_file)
            with open(tmp_file, mode="wb+") as buffer:
                copyfileobj(package.file, buffer)

            # logger.debug(f"Extracting files in {path}...")
            with self.extractor(tmp_file, mode=self.read_mode) as archive:
                # generate filenames and extensions
                filenames = [item for item in self._iterate(archive)]
                assert len(filenames) > 0, "Empty archive"
                # if there is more than one file, check for the presence of the ONNX file
                # and its config.pbtxt
                if len(filenames) > 1:
                    assert any(f.name == "config.pbtxt" for f in filenames), "Config file not found"
                    assert any(f.name.endswith(".onnx") for f in filenames), "ONNX file not found"
                    assert len(filenames) == 2, "Too many files in the archive"
                    # extract the config file
                    archive.extract("config.pbtxt", path=self.base_path / model.model_name)
                else:
                    # check that the file is called model.onnx
                    assert filenames[0].name == "model.onnx", "Missing 'model.onnx' file: either rename it "
                # extract the model
                archive.extract("model.onnx", model_path)

        return model_path

    def update(self, previous: ModelSchema, updated: ModelSchema) -> Path:
        destination = self.location(updated)
        move(self.location(previous), dst=destination)
        self._delete_empty_directories(self.base_path)
        return destination

    def delete(self, model: ModelSchema) -> None:
        rmtree(self.location(model), ignore_errors=False)
        self._delete_empty_directories(self.base_path)

    def close(self) -> None:
        self.base_path = None
