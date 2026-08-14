from collections.abc import Iterator
from pathlib import Path
from tarfile import TarFile
from zipfile import ZipFile

from triton_serve.storage import BaseExtractor


class ZipExtractor(BaseExtractor):
    archive: ZipFile

    def __enter__(self) -> "ZipExtractor":
        self.archive = ZipFile(self.file, "r")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.archive.close()

    def __iter__(self) -> Iterator[str]:
        return iter(self.archive.namelist())

    def extract(self, path: Path, member: str | None = None) -> None:
        if member is not None:
            self.archive.extract(member, path)
        else:
            self.archive.extractall(path)


class TarExtractor(BaseExtractor):
    archive: TarFile

    def __enter__(self) -> "TarExtractor":
        self.archive = TarFile.open(self.file, "r:gz")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.archive.close()

    def __iter__(self) -> Iterator[str]:
        return iter(self.archive.getnames())

    def extract(self, path: Path, member: str | None = None) -> None:
        if member is not None:
            self.archive.extract(member, path)
        else:
            self.archive.extractall(path)


ExtractorType = ZipExtractor | TarExtractor
