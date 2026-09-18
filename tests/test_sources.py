from pathlib import Path

import pytest

from triton_serve.storage.sources import RepositoryModelSource

MANIFEST = '[project]\nname = "b"\nversion = "0.1.0"\ndependencies = ["numpy==1.26.4"]\n'


def _bundle(root: Path) -> Path:
    (root / "model_repository" / "onnx" / "1").mkdir(parents=True)
    (root / "pyproject.toml").write_text(MANIFEST)
    return root


def test_repository_source_returns_models_and_dependencies(tmp_path, monkeypatch):
    source = RepositoryModelSource("git@example.org:acme/models", target_dir="model_repository")
    monkeypatch.setattr(source, "_fetch", lambda path: _bundle(path))

    bundle = source.extract(tmp_path)

    assert bundle.models == tmp_path / "model_repository"
    assert bundle.dependencies.pip == ["numpy==1.26.4"]


def test_repository_source_rejects_a_missing_models_directory(tmp_path, monkeypatch):
    source = RepositoryModelSource("git@example.org:acme/models", target_dir="model_repository")
    monkeypatch.setattr(source, "_fetch", lambda path: (path / "pyproject.toml").write_text(MANIFEST))

    with pytest.raises(AssertionError, match="model_repository"):
        source.extract(tmp_path)
