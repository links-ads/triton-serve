from collections.abc import Iterator
from tarfile import TarFile
from zipfile import ZipFile

from triton_serve.storage import BaseExtractor


class ZipExtractor(BaseExtractor[ZipFile]):
    def _open(self) -> ZipFile:
        return ZipFile(self.file, "r")

    def __iter__(self) -> Iterator[str]:
        return iter(self.archive.namelist())


class TarExtractor(BaseExtractor[TarFile]):
    def _open(self) -> TarFile:
        return TarFile.open(self.file, "r:gz")

    def __iter__(self) -> Iterator[str]:
        return iter(self.archive.getnames())


ExtractorType = ZipExtractor | TarExtractor
