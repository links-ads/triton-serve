from pathlib import Path

import pytest

from triton_serve.database.model import ModelType
from triton_serve.database.schema import ModelSchema, ModelVersionSchema
from triton_serve.storage import ModelStorageError


def make_model(name: str) -> ModelSchema:
    return ModelSchema(model_id=1, model_name=name, model_type=ModelType.ONNX)


def make_version(version_id: int) -> ModelVersionSchema:
    return ModelVersionSchema(model_id=1, version_id=version_id, model_uri="")


def make_origin(tmp_path: Path, name: str, version_id: int, payload: bytes = b"weights") -> Path:
    """Builds what a validated bundle hands to `save`, in its own directory so `save` may consume it."""
    origin = tmp_path / f"origin-{name}-{version_id}"
    version_dir = origin / name / str(version_id)
    version_dir.mkdir(parents=True)
    (version_dir / "model.onnx").write_bytes(payload)
    (origin / name / "config.pbtxt").write_text(f'name: "{name}"\n')
    return origin


def test_save_returns_the_location_it_wrote_to(storage, tmp_path):
    model, version = make_model("conf_save"), make_version(1)
    uri = storage.save(model, version, origin=make_origin(tmp_path, "conf_save", 1))
    assert uri == storage.location(model, version)


def test_a_second_version_joins_the_same_model(storage, tmp_path):
    model = make_model("conf_two")
    first = storage.save(model, make_version(1), origin=make_origin(tmp_path, "conf_two", 1))
    second = storage.save(model, make_version(2), origin=make_origin(tmp_path, "conf_two", 2))
    assert first != second
    assert Path(first).is_dir() and Path(second).is_dir()


def test_save_without_a_version_is_rejected(storage, tmp_path):
    origin = make_origin(tmp_path, "conf_missing", 1)
    with pytest.raises(ModelStorageError):
        storage.save(make_model("conf_missing"), make_version(7), origin=origin)


def test_delete_removes_the_version(storage, tmp_path):
    model, version = make_model("conf_del"), make_version(1)
    storage.save(model, version, origin=make_origin(tmp_path, "conf_del", 1))
    storage.save(model, make_version(2), origin=make_origin(tmp_path, "conf_del", 2))
    storage.delete(model, version)
    assert not Path(storage.location(model, version)).exists()
    assert Path(storage.location(model, make_version(2))).is_dir()


def test_deleting_a_version_that_is_not_there_is_refused(storage, tmp_path):
    model = make_model("conf_del_ghost")
    storage.save(model, make_version(1), origin=make_origin(tmp_path, "conf_del_ghost", 1))
    with pytest.raises(FileNotFoundError):
        storage.delete(model, make_version(9))


def test_rename_moves_every_version(storage, tmp_path):
    model = make_model("conf_from")
    storage.save(model, make_version(1), origin=make_origin(tmp_path, "conf_from", 1))
    storage.save(model, make_version(2), origin=make_origin(tmp_path, "conf_from", 2))

    storage.rename(model, "conf_to")

    renamed = make_model("conf_to")
    for version_id in (1, 2):
        assert Path(storage.location(renamed, make_version(version_id))).is_dir()
        assert not Path(storage.location(model, make_version(version_id))).exists()


def test_rename_onto_an_existing_name_is_refused(storage, tmp_path):
    source, occupied = make_model("conf_src"), make_model("conf_busy")
    storage.save(source, make_version(1), origin=make_origin(tmp_path, "conf_src", 1))
    storage.save(occupied, make_version(1), origin=make_origin(tmp_path, "conf_busy", 1))

    with pytest.raises(FileExistsError):
        storage.rename(source, "conf_busy")

    assert Path(storage.location(source, make_version(1))).is_dir()


def test_renaming_a_model_with_no_files_is_refused(storage):
    with pytest.raises(FileNotFoundError):
        storage.rename(make_model("conf_ghost"), "conf_ghost_2")


def test_stash_and_restore_round_trip(storage, tmp_path):
    model, version = make_model("conf_stash"), make_version(1)
    storage.save(model, version, origin=make_origin(tmp_path, "conf_stash", 1, payload=b"old"))

    stashed = storage.stash(model)
    assert not Path(storage.location(model, version)).exists()

    storage.restore(model, stashed)
    assert Path(storage.location(model, version) + "/model.onnx").read_bytes() == b"old"


def test_restore_replaces_whatever_the_failed_attempt_left_behind(storage, tmp_path):
    """The unwind path: a stash is taken, replacements are staged, then the transaction fails."""
    model, version = make_model("conf_replace"), make_version(1)
    storage.save(model, version, origin=make_origin(tmp_path, "conf_replace", 1, payload=b"old"))
    stashed = storage.stash(model)
    storage.save(model, version, origin=make_origin(tmp_path, "conf_replace", 1, payload=b"new"))

    storage.restore(model, stashed)

    assert Path(storage.location(model, version) + "/model.onnx").read_bytes() == b"old"


def test_a_second_stash_is_refused_while_one_is_outstanding(storage, tmp_path):
    """An occupied stash means a previous transaction died, and its files may be the only copy."""
    model = make_model("conf_busy_stash")
    storage.save(model, make_version(1), origin=make_origin(tmp_path, "conf_busy_stash", 1))
    storage.stash(model)
    storage.save(model, make_version(1), origin=make_origin(tmp_path, "conf_busy_stash", 1))

    with pytest.raises(FileExistsError):
        storage.stash(model)


def test_discard_drops_the_stash(storage, tmp_path):
    model = make_model("conf_discard")
    storage.save(model, make_version(1), origin=make_origin(tmp_path, "conf_discard", 1))
    stashed = storage.stash(model)

    storage.discard(stashed)

    assert not Path(stashed).exists()


def test_discard_of_an_already_gone_stash_is_quiet(storage, tmp_path):
    """`discard` runs on the success path; it must never be the thing that fails a commit."""
    model = make_model("conf_twice")
    storage.save(model, make_version(1), origin=make_origin(tmp_path, "conf_twice", 1))
    stashed = storage.stash(model)
    storage.discard(stashed)
    storage.discard(stashed)
