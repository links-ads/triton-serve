import logging
import tempfile
from pathlib import Path

from fastapi import HTTPException
from sqlalchemy.orm import Session

from triton_serve.api.dto import ModelUpdateBody
from triton_serve.builder.execute import enqueue_build
from triton_serve.builder.resolve import resolve_service_image, services_using_models
from triton_serve.config.schema import AppSettings
from triton_serve.database.model import Model, ModelVersion, timezone_aware_now
from triton_serve.database.schema import ModelCreateSchema
from triton_serve.storage import ModelSource, ModelStorage
from triton_serve.storage.validation import validate_models

LOG = logging.getLogger("uvicorn")


def refresh_service_images(db: Session, models: list[Model], settings: AppSettings) -> None:
    """Re-resolves every live service serving any of these models, and queues the builds it needs.

    A changed requirements.txt changes the dependency union, which changes the content hash, which
    is a different image. Fan-out is eager: at this scale the affected set is a handful of services.

    Args:
        db (Session): The database session.
        models (list[Model]): The models just created or updated.
        settings (AppSettings): The application settings.

    Raises:
        HTTPException: 422 if a stored dependency does not parse.
    """
    pending: list[str] = []
    for service in services_using_models(db, models):
        try:
            if (image_hash := resolve_service_image(db=db, service=service, settings=settings)) is not None:
                pending.append(image_hash)
        except ValueError as e:
            db.rollback()
            raise HTTPException(
                status_code=422, detail=f"Invalid dependencies for '{service.service_name}': {e}"
            ) from e
    db.commit()
    for image_hash in pending:
        enqueue_build(image_hash)


def get_single_model(
    db: Session,
    model_name: str,
) -> Model | None:
    """
    Retrieves a model given a unique name.

    Args:
        db (Session): The database session.
        storage (ModelStorage): The storage implementation to use.
        model_name (str): The name of the model to retrieve.

    Returns:
        Model: Returns the requested model instance if found, or None otherwise.
    """
    model = (
        db.query(Model)
        .filter(
            Model.model_name == model_name,
            Model.deleted_at.is_(None),
        )
        .first()
    )
    return model


def get_all_models(
    db: Session,
    model_name: str | None = None,
    deleted: bool = False,
    source: str | None = None,
) -> list[Model]:
    """
    Retrieves a list of models filtered by the given parameters, if provided.

    Args:
        db (Session): The database session to query from.
        model_name (Optional[str], optional): The name of the model to filter. Defaults to None.
        deleted (bool, optional): Whether to include deleted models. Defaults to False.
        source (Optional[str], optional): The source of the model to filter. Defaults to None.

    Returns:
        List[ModelSchema]: A list of ModelSchema instances representing the filtered models.
    """
    statement = db.query(Model)
    if model_name is not None:
        statement = statement.filter(Model.model_name == model_name)
    if not deleted:
        statement = statement.filter(Model.deleted_at.is_(None))
    if source is not None:
        if not source.startswith("git@") and not source.endswith((".zip", ".tgz", ".tar.gz")):
            raise HTTPException(
                status_code=500, detail="Invalid source format. Must be a git repository or an archive file."
            )
        statement = statement.filter(Model.source == source)
    return statement.all()


def create_models_from_source(
    source: ModelSource,
    storage: ModelStorage,
    db: Session,
    update: bool = False,
) -> list[Model]:
    """
    Extracts models from a source archive and creates them in the database.

    A bundle registers as one unit: the whole set commits once, at the end, so a model that fails
    halfway through does not leave the ones before it registered. Storage is not part of that
    transaction -- files already moved for the earlier models stay on disk, to be overwritten by
    the next attempt at the same bundle.

    Args:
        source (ModelSource): The source of the models to extract, either archive or git repository.
        storage (ModelStorage): The storage implementation to use.
        db (Session): The database session.
        update (bool, optional): Whether to update the models if they already exist. Defaults to False.

    Returns:
        List[Model]: A list of Model instances representing the extracted models.

    Raises:
        HTTPException: If the file is invalid.
    """
    try:
        models = []
        with tempfile.TemporaryDirectory() as tmp_dir:
            models_origin = source.origin()
            tmp_repository = source.extract(path=Path(tmp_dir))
            validated_models: list[ModelCreateSchema] = validate_models(tmp_repository)
            # store the models in the database
            for instance in validated_models:
                # verify the model is not already in the database
                if old_model := get_single_model(db=db, model_name=instance.model_name):
                    if not update:
                        raise HTTPException(
                            status_code=409,
                            detail=f"Model '{instance.model_name}' already exists",
                        )
                    # if update is enabled, update the model fields...
                    # first check if the model has the same source
                    if old_model.source != models_origin:
                        raise ValueError(
                            f"Old model source '{old_model.source}' does not match new source '{models_origin}' for '{instance.model_name}'."
                            "If the new source is correct, delete the model and re-register it with the correct source."
                        )
                    old_model.model_type = instance.model_type
                    old_model.source = instance.source or models_origin
                    old_model.dependencies = instance.dependencies  # type: ignore
                    old_model.system_dependencies = instance.system_dependencies  # type: ignore
                    old_model.version_policy = instance.version_policy  # type: ignore
                    old_model.updated_at = timezone_aware_now()
                    # clean up the versions
                    for version in old_model.versions:
                        storage.delete(old_model, version)
                        db.delete(version)
                    old_model.versions = []
                    # ... then update its versions
                    for version in instance.versions:
                        version.model_id = old_model.model_id  # type: ignore
                        version.model_uri = str(storage.save(old_model, version, origin=tmp_repository))  # type: ignore
                        old_model.versions.append(ModelVersion(**version.model_dump()))
                    model = old_model

                # if the model does not exist, create a new model
                # store files in the repository and create a new model instance
                else:
                    model_versions = []
                    instance.source = instance.source or models_origin
                    for version in instance.versions:
                        version.model_uri = str(storage.save(instance, version, origin=tmp_repository))  # type: ignore
                        model_versions.append(ModelVersion(**version.model_dump()))

                    model = Model(**{**instance.model_dump(), "versions": model_versions})
                    db.add(model)

                models.append(model)

            db.commit()
            for model in models:
                db.refresh(model)

        return models
    except HTTPException:
        # a conflict raised mid-loop must discard the models already staged before it
        db.rollback()
        raise
    except (AssertionError, ValueError) as e:
        db.rollback()
        raise HTTPException(status_code=422, detail=f"Cannot register model(s): {e}") from e


def edit_model_info(db: Session, storage: ModelStorage, model: Model, updates: ModelUpdateBody) -> Model:
    """
    Updates a model given the name and the version.
    If the name or the version are provided in the updates, update the model and move the model to the new location.

    Args:
        storage (ModelStorage): The storage implementation to use.
        model (ModelSchema): The model to update.
        updates (ModelUpdateBody): The updates to apply.

    Raises:
        HTTPException: If the model could not be updated.

    Returns:
        ModelSchema: The updated model.

    """
    try:
        updated_name = updates.name or model.model_name
        # check if the updated model exists
        assert get_single_model(db=db, model_name=updated_name) is None, f"Model '{updated_name}' already exists"
        # update the model
        model.model_name = updated_name
        model.source = updates.source or model.source
        model.updated_at = timezone_aware_now()
        for version in model.versions:
            version.model_uri = str(storage.update(model, version, current_uri=Path(version.model_uri)))
        db.commit()
        db.refresh(model)
        return model
    except AssertionError as e:
        raise HTTPException(status_code=409, detail=f"Cannot update model: {e}") from e
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Cannot update model: {e}") from e


def delete_model(
    db: Session,
    storage: ModelStorage,
    model: Model,
    version_number: int | None,
) -> Model:
    """
    Deletes a model given the name and the version.

    Args:
        db (Session): The database session.
        storage (ModelStorage): The storage implementation to use.
        model (ModelSchema): The model to delete.
        version_number (int): The version of the model to delete.

    Raises:
        HTTPException: If the model could not be deleted.

    Returns:
        None

    """
    LOG.debug("Deleting model '%s' (version: %s)", model.model_name, version_number)
    if version_number is not None:
        model_version = db.query(ModelVersion).get((model.model_id, version_number))
        if model_version:
            storage.delete(model, model_version)
            db.delete(model_version)
            db.flush()
    else:
        for model_version in model.versions:
            storage.delete(model, model_version)
            db.delete(model_version)
        db.flush()

    # check if the model has any versions left
    remaining = db.query(ModelVersion).filter(ModelVersion.model_id == model.model_id).first()
    if remaining is None:
        model.deleted_at = timezone_aware_now()

    db.commit()
    db.refresh(model)
    return model
