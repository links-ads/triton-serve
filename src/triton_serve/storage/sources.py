import shutil
import subprocess
from pathlib import Path

from fastapi import UploadFile

from triton_serve.storage import ModelSource
from triton_serve.storage.extractors import ExtractorType, TarExtractor, ZipExtractor


class ArchiveModelSource(ModelSource):
    """Model source that extracts models from an archive file."""

    def __init__(self, package: UploadFile, target_dir: str = None):
        self.package = package
        self.package_name = package.filename
        self.target_dir = target_dir or "model_repository"

    def _get_extractor(self, filename: Path) -> ExtractorType:
        """Checks the extension from the filename, returning the correct
        extractor implementation.

        Args:
            filename (Path): Path to the file.

        Raises:
            ValueError: If the file format is not supported.

        Returns:
            ExtractorType: Extractor implementation.
        """
        if filename.endswith(".zip"):
            return ZipExtractor
        elif filename.endswith(".tar.gz") or filename.endswith(".tgz"):
            return TarExtractor
        else:
            suffix = Path(filename).suffix
            raise ValueError(f"Unsupported file format: {suffix}")

    def origin(self) -> str:
        return self.package_name

    def extract(self, path: Path):
        extractor = self._get_extractor(filename=self.package.filename)
        temp_file = path / self.package.filename
        # sto the upload file in a temporary file
        with open(temp_file, mode="wb+") as buffer:
            shutil.copyfileobj(self.package.file, buffer)

        # do a preliminary check on the archive, to see if it's empty
        # or to see if it contains the expected structure
        with extractor(temp_file) as archive:
            filenames = {item for item in archive}
            # check if the archive is not empty and contains the expected structure
            assert len(filenames) > 0, "Empty archive"
            assert all([item.startswith(self.target_dir) for item in filenames]), "Invalid archive structure"
            # extract everything: we validate its content later
            archive.extract(path)

        temp_file.unlink()
        self.package = None
        return path / self.target_dir


class RepositoryModelSource(ModelSource):
    """Model source that extracts models from a git repository."""

    def __init__(self, url: str, target_dir: str = None):
        self.url = url
        self.target_dir = target_dir or "model_repository"

    def origin(self) -> str:
        return self.url

    def extract(self, path: Path) -> Path:
        """Clones the git repository and pulls LFS files.

        Args:
            path (Path): Path where to clone the repository.

        Returns:
            Path: Path to the extracted model repository.
        """
        target_path = path / self.target_dir
        subprocess.run(["git", "clone", self.url, str(path)], check=True)
        subprocess.run(["git", "lfs", "pull"], cwd=path, check=True)
        subprocess.run(["rm", "-rf", str(path / ".git")], check=True)
        return target_path
