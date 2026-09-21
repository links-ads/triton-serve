from contextlib import suppress
from pathlib import Path
from shutil import move, rmtree

from triton_serve.storage.base import ModelStorage, StorableModel, StorableVersion, StorageURI

STASH_DIRNAME = ".stash"


class LocalModelStorage(ModelStorage):
    """Storage backed by a local filesystem.

    The stash is a reserved directory inside the repository rather than beside it, so every
    move-aside is a single atomic rename instead of a cross-filesystem copy. This is safe as long
    as the worker image's entrypoint (`serve-triton`) keeps starting Triton in explicit
    model-control mode, where only the names it is told to load are looked at.
    """

    def __init__(self, base_path: Path) -> None:
        self.base_path = base_path

    def _model_root(self, model_name: str) -> Path:
        return self.base_path / model_name

    def _stash_root(self, model_name: str) -> Path:
        return self.base_path / STASH_DIRNAME / model_name

    def location(self, model: StorableModel, version: StorableVersion) -> StorageURI:
        return str(self._model_root(model.model_name) / str(version.version_id))

    def save(self, model: StorableModel, version: StorableVersion, origin: Path) -> StorageURI:
        model_root = self._model_root(model.model_name)
        model_root.mkdir(parents=True, exist_ok=True)
        config_tmp = origin / model.model_name / "config.pbtxt"
        version_tmp = origin / model.model_name / str(version.version_id)
        assert version_tmp.exists(), f"Version {model.model_name}:{version.version_id} does not exist"
        # a bundle that omits the config only registers against a model that already has one
        if config_tmp.exists():
            move(config_tmp, dst=model_root / "config.pbtxt")
        else:
            assert (model_root / "config.pbtxt").exists(), f"Missing config file in {model.model_name}"
        move(version_tmp, dst=model_root)
        return self.location(model, version)

    def delete(self, model: StorableModel, version: StorableVersion) -> None:
        rmtree(self.location(model, version), ignore_errors=False)
        model_root = self._model_root(model.model_name)
        if not any(child.is_dir() for child in model_root.iterdir()):
            rmtree(model_root, ignore_errors=False)

    def rename(self, model: StorableModel, new_name: str) -> None:
        source = self._model_root(model.model_name)
        destination = self._model_root(new_name)
        if not source.exists():
            raise FileNotFoundError(f"Model '{model.model_name}' has no files at {source}")
        if destination.exists():
            raise FileExistsError(f"Model '{new_name}' already has files at {destination}")
        source.rename(destination)

    def stash(self, model: StorableModel) -> StorageURI:
        source = self._model_root(model.model_name)
        destination = self._stash_root(model.model_name)
        if not source.exists():
            raise FileNotFoundError(f"Model '{model.model_name}' has no files at {source}")
        if destination.exists():
            raise FileExistsError(f"A stash is already outstanding for '{model.model_name}' at {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        source.rename(destination)
        return str(destination)

    def restore(self, model: StorableModel, stashed: StorageURI) -> None:
        source = Path(stashed)
        if not source.exists():
            raise FileNotFoundError(f"No stash at {stashed}")
        destination = self._model_root(model.model_name)
        # whatever sits here was written by the attempt that failed, and the stash predates it
        if destination.exists():
            rmtree(destination, ignore_errors=False)
        source.rename(destination)

    def discard(self, stashed: StorageURI) -> None:
        with suppress(FileNotFoundError):
            rmtree(stashed, ignore_errors=False)
