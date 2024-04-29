from pathlib import Path
from shutil import copy, move, rmtree

from triton_serve.database.schema import ModelSchema
from triton_serve.storage import ModelStorage


class LocalModelStorage(ModelStorage):
    """Storage implementation based on a local file system.
    This module handles a single path, managing file upload and extraction, update and deletion.
    """

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

    def _contains_directory(self, path: Path, name: str = None) -> bool:
        """Returns True if the given path contains a directory with the given name.

        Args:
            path (Path): The path to search for directories.
            name (str, optional): Optional directory name. Defaults to None.

        Returns:
            bool: true if present (and name matches, if provided), false otherwise.
        """
        for item in path.iterdir():
            if item.is_dir():
                if name is None or item.name == name:
                    return True
        return False

    def load(self, model: ModelSchema) -> str:
        """Simply returns the model URI.

        Args:
            model (ModelSchema): The model to load.

        Returns:
            str: The model URI.
        """
        return self.location(model)

    def exists(self, model: ModelSchema) -> bool:
        """Checks if the model exists in the base path.

        Args:
            model (ModelSchema): The model to check.

        Returns:
            bool: True if the model exists, False otherwise.
        """
        return Path(self.location(model)).exists()

    def save(self, model: ModelSchema, origin: Path) -> Path:
        """
        Moves the model from the origin to the base path.
        There are two possible scenarios:
        - The model is new and is being uploaded for the first time: model_repository/model_name does not exist.
        - The model already exists and a new version is being uploaded: the folder model_repository/model_name exists.
        In the first case, the model is moved from the temporary directory to the base path.
        In the second case, only the specific version folder is moved to the base path,
        and the config.pbtxt file is updated with the new version, if present.

        Args:
            model (ModelSchema): The model to save.
            origin (Path): The path to the temporary directory containing the model.

        Returns:
            Path: The model URI.

        Raises:
            AssertionError: If the version folder does not exist.
        """
        model_uri = self.location(model)
        model_root = Path(model_uri).parent
        # create the model directory if it does not exist
        model_root.mkdir(exist_ok=True)
        # compose the temporary paths for the config file and the version directory
        config_tmp = origin / model.model_name / "config.pbtxt"
        model_version_tmp = origin / model.model_name / str(model.model_version)
        # check if the version directory exists, this is a must
        assert model_version_tmp.exists(), f"Version {model.model_name}:{model.model_version} does not exist"
        # if the config file is present, move it to the model root
        # using the full path in destination overwrites the original file
        if config_tmp.exists():
            move(config_tmp, dst=model_root / "config.pbtxt")
        # if it is not present, check if the model root already has a config file
        else:
            assert (model_root / "config.pbtxt").exists(), f"Missing config file in {model.model_name}"
        # move the version directory to the model root
        move(model_version_tmp, dst=model_root)
        # return the model URI
        return model_uri

    def update(self, updated: ModelSchema, current_uri: Path) -> Path:
        """
        Moves the updated model to the base path.
        The model is moved from the temporary directory to the base path.
        The model URI is updated with the new version.

        Args:
            updated (ModelSchema): The updated model.
            origin (Path): The path to the temporary directory containing the model.

        Returns:
            Path: The model URI.
        """
        # two cases again: the new name does not exist, or it does
        updated_uri = self.location(updated)
        updated_root = Path(updated_uri).parent
        current_root = current_uri.parent
        print(f"current_root: {current_root}")
        print(f"updated_root: {updated_root}")
        print(f"current_uri: {current_uri}")
        print(f"updated_uri: {updated_uri}")
        # case 1 - name does not exist: create the model directory and move the version directory inside
        # also add the config file from the original model
        if not updated_root.exists():
            print("Root does not exist")
            updated_root.mkdir()
            move(current_uri, dst=updated_uri)
            config_file = current_root / "config.pbtxt"
            assert config_file.exists(), f"Missing config file in {current_uri.parent.name}"
            # check if the original model has no other versions: if so, move the config file
            # otherwise copy it to the new model directory
            if not self._contains_directory(current_root):
                move(config_file, dst=updated_root / "config.pbtxt")
            else:
                copy(config_file, dst=updated_root / "config.pbtxt")
        # case 2 - the model already exists, we just need to move to the version
        else:
            print("Root exists")
            move(current_uri, dst=updated_uri)
        # return the updated model URI
        # extra check to remove empty directories, if any
        self._delete_empty_directories(self.base_path)
        return updated_uri

    def delete(self, model: ModelSchema) -> None:
        """
        Deletes the model from the base path. If the model is the last one in the directory,
        the directory is removed as well.
        """
        model_uri = self.location(model)
        rmtree(model_uri, ignore_errors=False)
        model_root = Path(self.location(model)).parent
        # if the model root does not have version folders, remove it
        if not self._contains_directory(model_root):
            rmtree(model_root, ignore_errors=False)
        # extra check to remove empty directories, if any
        self._delete_empty_directories(self.base_path)

    def close(self) -> None:
        """Not much to do here."""
        self.base_path = None
