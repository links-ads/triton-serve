import os
from datetime import timedelta

import pytest

from triton_serve import tasks
from triton_serve.database.model import Model, ModelType, ModelVersion, timezone_aware_now
from triton_serve.storage.base import StoredVersion
from triton_serve.storage.local import LocalModelStorage


def _stored_version(root, name: str, version_id: int, age: int):
    """Writes a version directory and backdates it, so the sweep sees it as that old."""
    version_dir = root / name / str(version_id)
    version_dir.mkdir(parents=True)
    (version_dir / "model.onnx").write_bytes(b"weights")
    (root / name / "config.pbtxt").write_text("")
    old = (timezone_aware_now() - timedelta(seconds=age)).timestamp()
    os.utime(version_dir, (old, old))
    return version_dir


@pytest.fixture
def repository(tmp_path, monkeypatch):
    root = tmp_path / "models"
    root.mkdir()
    monkeypatch.setattr(tasks, "get_storage", lambda: LocalModelStorage(root))
    return root


@pytest.fixture
def registered(test_db):
    """One real row, so the sweep's zero-rows guard does not fire during the other cases."""
    model = Model(
        model_name="sweep_known",
        model_type=ModelType.ONNX,
        source="bundle.zip",
        dependencies=[],
        system_dependencies=[],
        versions=[ModelVersion(version_id=1, model_uri="unused")],
    )
    test_db.add(model)
    test_db.commit()
    yield model
    test_db.query(ModelVersion).filter(ModelVersion.model_id == model.model_id).delete()
    test_db.query(Model).filter(Model.model_id == model.model_id).delete()
    test_db.commit()


def test_the_sweep_deletes_an_orphan_past_the_grace_period(repository, registered, test_settings):
    orphan = _stored_version(repository, "sweep_ghost", 1, age=test_settings.orphan_min_age + 60)

    tasks.sweep_orphaned_files()

    assert not orphan.exists()


def test_the_sweep_leaves_a_young_orphan_alone(repository, registered, test_settings):
    """A registration stages files before it commits rows, so a fresh directory may not be an orphan."""
    fresh = _stored_version(repository, "sweep_fresh", 1, age=0)

    tasks.sweep_orphaned_files()

    assert fresh.exists()


def test_the_sweep_leaves_a_registered_version_alone(repository, registered, test_settings):
    kept = _stored_version(repository, "sweep_known", 1, age=test_settings.orphan_min_age + 60)

    tasks.sweep_orphaned_files()

    assert kept.exists()


def test_no_database_rows_means_no_orphans():
    """An empty database beside a full repository is an outage, never a mandate to delete everything."""
    cutoff = timezone_aware_now()
    stored = [StoredVersion("anything", 1, cutoff - timedelta(days=30))]

    assert tasks._orphans(stored, known=set(), cutoff=cutoff) == []
