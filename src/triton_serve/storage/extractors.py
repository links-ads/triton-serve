from pathlib import Path

from triton_serve.storage import BaseExtractor


class ZipExtractor(BaseExtractor):
    def __enter__(self, mode: str = "r"):
        from zipfile import ZipFile

        self.archive = ZipFile(self.file, mode=mode)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.archive.close()

    def __iter__(self):
        return iter(self.archive.namelist())

    def extract(self, path: Path, member: str = None):
        if member is not None:
            self.archive.extract(member, path)
        else:
            self.archive.extractall(path)


class TarExtractor(BaseExtractor):
    def __enter__(self, mode: str = "r:gz"):
        from tarfile import TarFile

        self.archive = TarFile.open(self.file, mode=mode)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.archive.close()

    def __iter__(self):
        return iter(self.archive.getnames())

    def extract(self, path: Path, member: str = None):
        if member is not None:
            self.archive.extract(member, path)
        else:
            self.archive.extractall(path)


ExtractorType = ZipExtractor | TarExtractor
