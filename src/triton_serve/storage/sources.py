import shutil
import subprocess
from pathlib import Path

from fastapi import UploadFile

from triton_serve.storage.base import ExtractedBundle, ModelSource
from triton_serve.storage.extractors import ExtractorType, TarExtractor, ZipExtractor
from triton_serve.storage.validation import parse_dependencies


class ArchiveModelSource(ModelSource):
    """Model source that extracts models from an archive file."""

    def __init__(self, package: UploadFile, target_dir: str):
        super().__init__(target_dir)
        self.package = package
        assert package.filename is not None, "Invalid package name"
        self.package_name: str = package.filename

    def _get_extractor(self, filename: Path | str) -> type[ExtractorType]:
        """Checks the extension from the filename, returning the correct
        extractor implementation.

        Args:
            filename (Path): Path to the file.

        Raises:
            ValueError: If the file format is not supported.

        Returns:
            type[ExtractorType]: Extractor implementation.
        """
        if str(filename).endswith(".zip"):
            return ZipExtractor
        elif str(filename).endswith(".tar.gz") or str(filename).endswith(".tgz"):
            return TarExtractor
        else:
            suffix = Path(filename).suffix
            raise ValueError(f"Unsupported file format: {suffix}")

    def origin(self) -> str:
        return self.package_name

    def extract(self, path: Path) -> ExtractedBundle:
        """Unpacks the archive and reads the bundle's manifest.

        Args:
            path (Path): Path where to unpack the archive.

        Returns:
            ExtractedBundle: The models directory and the bundle's declared dependencies.

        Raises:
            AssertionError: If the archive is not a valid bundle.
        """
        assert self.package is not None, "Missing package"
        extractor = self._get_extractor(filename=self.package_name)
        temp_file = path / self.package_name
        # sto the upload file in a temporary file
        with open(temp_file, mode="wb+") as buffer:
            shutil.copyfileobj(self.package.file, buffer)

        # the structure is checked on the member list, before anything is written to disk
        with extractor(temp_file) as archive:
            members = {item for item in archive}
            assert members, "Empty archive"
            assert any(item.startswith(f"{self.target_dir}/") for item in members), (
                f"Invalid archive structure: no {self.target_dir}/ directory"
            )
            assert "pyproject.toml" in members, "Invalid archive structure: missing pyproject.toml at the archive root"
            # extract everything: we validate its content later
            archive.extract(path)

        temp_file.unlink()
        self.package = None
        return ExtractedBundle(
            models=path / self.target_dir,
            dependencies=parse_dependencies(path, self.target_dir),
        )


class RepositoryModelSource(ModelSource):
    """Model source that extracts models from a git repository."""

    def __init__(self, url: str, target_dir: str):
        # check that the URL is a valid git SSH URL
        assert url.startswith("git@"), "Invalid git URL, use SSH format (git@...)"
        super().__init__(target_dir)
        self.url = url

    def origin(self) -> str:
        return self.url

    def _fetch(self, path: Path) -> None:
        subprocess.run(["git", "clone", self.url, str(path)], check=True)
        subprocess.run(["git", "lfs", "pull"], cwd=path, check=True)
        subprocess.run(["rm", "-rf", str(path / ".git")], check=True)

    def extract(self, path: Path) -> ExtractedBundle:
        """Clones the repository, pulls LFS files, and reads the bundle's manifest.

        Args:
            path (Path): Path where to clone the repository.

        Returns:
            ExtractedBundle: The models directory and the bundle's declared dependencies.

        Raises:
            AssertionError: If the clone is not a valid bundle.
        """
        self._fetch(path)
        models = path / self.target_dir
        assert models.is_dir(), f"Invalid repository structure: missing {self.target_dir}/ directory"
        return ExtractedBundle(models=models, dependencies=parse_dependencies(path, self.target_dir))
