from pathlib import Path

import pytest

from triton_serve.database.model import ModelType
from triton_serve.database.schema import ModelSchema, ModelVersionSchema
from triton_serve.storage import ModelStorageError
from triton_serve.storage.local import LocalModelStorage


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
    assert storage.exists(first) and storage.exists(second)


def test_save_without_a_version_is_rejected(storage, tmp_path):
    origin = make_origin(tmp_path, "conf_missing", 1)
    with pytest.raises(ModelStorageError):
        storage.save(make_model("conf_missing"), make_version(7), origin=origin)


def test_delete_removes_the_version(storage, tmp_path):
    model, version = make_model("conf_del"), make_version(1)
    storage.save(model, version, origin=make_origin(tmp_path, "conf_del", 1))
    storage.save(model, make_version(2), origin=make_origin(tmp_path, "conf_del", 2))
    storage.delete(model, version)
    assert not storage.exists(storage.location(model, version))
    assert storage.exists(storage.location(model, make_version(2)))


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
        assert storage.exists(storage.location(renamed, make_version(version_id)))
        assert not storage.exists(storage.location(model, make_version(version_id)))


def test_rename_onto_an_existing_name_is_refused(storage, tmp_path):
    source, occupied = make_model("conf_src"), make_model("conf_busy")
    storage.save(source, make_version(1), origin=make_origin(tmp_path, "conf_src", 1))
    storage.save(occupied, make_version(1), origin=make_origin(tmp_path, "conf_busy", 1))

    with pytest.raises(FileExistsError):
        storage.rename(source, "conf_busy")

    assert storage.exists(storage.location(source, make_version(1)))


def test_renaming_a_model_with_no_files_is_refused(storage):
    with pytest.raises(FileNotFoundError):
        storage.rename(make_model("conf_ghost"), "conf_ghost_2")


def test_stash_and_restore_round_trip(storage, tmp_path):
    model, version = make_model("conf_stash"), make_version(1)
    storage.save(model, version, origin=make_origin(tmp_path, "conf_stash", 1, payload=b"old"))

    stashed = storage.stash(model)
    assert not storage.exists(storage.location(model, version))

    storage.restore(model, stashed)
    assert storage.read(storage.location(model, version) + "/model.onnx") == b"old"


def test_restore_replaces_whatever_the_failed_attempt_left_behind(storage, tmp_path):
    """The unwind path: a stash is taken, replacements are staged, then the transaction fails."""
    model, version = make_model("conf_replace"), make_version(1)
    storage.save(model, version, origin=make_origin(tmp_path, "conf_replace", 1, payload=b"old"))
    stashed = storage.stash(model)
    storage.save(model, version, origin=make_origin(tmp_path, "conf_replace", 1, payload=b"new"))

    storage.restore(model, stashed)

    assert storage.read(storage.location(model, version) + "/model.onnx") == b"old"


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

    assert not storage.exists(stashed)


def test_discard_of_an_already_gone_stash_is_quiet(storage, tmp_path):
    """`discard` runs on the success path; it must never be the thing that fails a commit."""
    model = make_model("conf_twice")
    storage.save(model, make_version(1), origin=make_origin(tmp_path, "conf_twice", 1))
    stashed = storage.stash(model)
    storage.discard(stashed)
    storage.discard(stashed)


def test_reading_something_that_is_not_there_raises_the_same_error_everywhere(storage):
    model, version = make_model("conf_read_ghost"), make_version(1)
    with pytest.raises(FileNotFoundError):
        storage.read(storage.location(model, version))


def test_a_reachable_backend_passes_its_own_check(storage):
    assert storage.check_reachable() is None


def test_a_local_repository_that_is_not_there_is_refused(tmp_path):
    backend = LocalModelStorage(tmp_path / "absent")
    with pytest.raises(FileNotFoundError, match="absent"):
        backend.check_reachable()


def test_an_azure_container_that_does_not_exist_is_refused(storage):
    """Points a second backend at a container the fixture never created, on the same Azurite."""
    if type(storage).__name__ != "AzureModelStorage":
        pytest.skip("the local backend has no container to miss")
    backend = type(storage)(
        account=storage.account,
        container="never-created",
        credential=storage._credential,
        prefix="models",
        # BlobServiceClient.url is the account url with a trailing slash, which the sdk normalises
        endpoint=storage.client.url,
    )
    with pytest.raises(ConnectionError, match="never-created"):
        backend.check_reachable()


def test_the_local_backend_mounts_the_repository(tmp_path):
    repository = tmp_path / "models"
    repository.mkdir()
    wiring = LocalModelStorage(repository, volume="serve-test_models").worker_repository()

    assert wiring.uri == "/models"
    assert wiring.mounts == {"serve-test_models": {"bind": "/models", "mode": "ro"}}
    assert wiring.environment == {"WORKER_REPOSITORY": "/models"}


def test_the_local_backend_mounts_nothing_without_a_volume(tmp_path):
    repository = tmp_path / "models"
    repository.mkdir()
    wiring = LocalModelStorage(repository).worker_repository()

    assert wiring.mounts == {}


def _azure_backend(**kwargs):
    from triton_serve.storage.azure import AzureModelStorage

    return AzureModelStorage(
        **{
            "account": "adsmodelrepository",
            "container": "model-repository",
            "credential": "deadbeef",
            "prefix": "models",
            **kwargs,
        }
    )


def test_the_azure_backend_mounts_nothing_and_carries_its_credentials():
    pytest.importorskip("azure.storage.blob")
    wiring = _azure_backend().worker_repository()

    assert wiring.uri == "as://adsmodelrepository/model-repository/models"
    assert wiring.mounts == {}
    assert wiring.environment["WORKER_REPOSITORY"] == wiring.uri
    assert wiring.environment["AZURE_STORAGE_KEY"] == "deadbeef"


@pytest.mark.parametrize("prefix", ["models", "team/models"])
def test_the_azure_stash_stays_outside_what_the_workers_can_see(prefix):
    """A stash under the repository prefix would be loaded as a model named `.stash`."""
    pytest.importorskip("azure.storage.blob")
    backend = _azure_backend(prefix=prefix)

    stashed = backend._stash_key("resnet")

    assert not stashed.startswith(f"{backend.prefix}/")
    assert stashed == ".stash/resnet"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"prefix": ""},
        {"prefix": "/"},
        {"stash_prefix": ""},
        {"prefix": "models", "stash_prefix": "models"},
        {"prefix": "models", "stash_prefix": "models/.stash"},
    ],
)
def test_an_azure_layout_that_exposes_the_stash_is_refused(kwargs):
    pytest.importorskip("azure.storage.blob")
    with pytest.raises(ValueError):
        _azure_backend(**kwargs)


def test_storage_wiring_wins_over_a_user_supplied_environment():
    """A service creator must not be able to repoint a worker at a repository of their choosing."""
    from triton_serve.api.services.domain import merge_environment

    merged = merge_environment(
        {"WORKER_REPOSITORY": "/somewhere/else", "MY_FLAG": "1"},
        {"WORKER_REPOSITORY": "/models"},
    )

    assert merged == {"MY_FLAG": "1", "WORKER_REPOSITORY": "/models"}
