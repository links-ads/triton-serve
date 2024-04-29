import logging
import tempfile
from pathlib import Path

from fastapi import HTTPException
from sqlalchemy.orm import Session

from triton_serve.api.dto import ModelUpdateBody
from triton_serve.database.model import Model
from triton_serve.database.schema import timezone_aware_now
from triton_serve.storage import ModelSource, ModelStorage
from triton_serve.storage.validation import validate_models

LOG = logging.getLogger("uvicorn")


def get_model(db: Session, storage: ModelStorage, model_name: str, model_version: int) -> Model | None:
    """
    Retrieves a model given a unique combination of name and version.

    Args:
        db (Session): The database session.
        storage (ModelStorage): The storage implementation to use.
        model_name (str): The name of the model to retrieve.
        model_version (int): The version of the model to retrieve.

    Returns:
        ModelSchema: Returns the requested ModelSchema instance if found, or None otherwise.
    """
    LOG.debug(f"Retrieving model {model_name}:{model_version}")
    model = db.query(Model).filter(Model.model_name == model_name, Model.model_version == model_version).first()
    if model is not None:
        assert storage.exists(model), f"Model URI {model.model_uri} does not exist"
    return model


def list_models(
    db: Session,
    storage: ModelStorage,
    model_name: str | None = None,
    version: int | None = None,
) -> list[Model]:
    """
    Retrieves a list of models filtered by the given parameters, if provided.

    Args:
        repository_path (Path): The path to the directory containing the models.
        model_name (Optional[str], optional): The name of the model to filter. Defaults to None.
        version (Optional[int], optional): The version of the model to filter. Defaults to None.

    Returns:
        List[ModelSchema]: A list of ModelSchema instances representing the filtered models.
    """
    # query models from database based on the given parameters
    LOG.debug(f"Retrieving models with name {model_name} and version {version}")
    statement = db.query(Model)
    if model_name is not None:
        statement = statement.filter(Model.model_name == model_name)
    if version is not None:
        statement = statement.filter(Model.model_version == version)
    models = statement.all()
    # assert the stored path exists
    for model in models:
        assert storage.exists(model), f"Model URI {model.model_uri} does not exist"

    # return the list of models
    return models


def create_models_from_source(
    source: ModelSource,
    storage: ModelStorage,
    db: Session,
    update: bool = False,
) -> list[Model]:
    """
    Extracts models from a source archive and creates them in the database.

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
            schemas = validate_models(tmp_repository)
            # store the models in the database
            for schema in schemas:
                # verify the model is not already in the database
                LOG.debug(f"Creating model {schema.model_name}:{schema.model_version}")
                if model := get_model(
                    db=db,
                    storage=storage,
                    model_name=schema.model_name,
                    model_version=schema.model_version,
                ):
                    # delete the old model if update is enabled
                    if update:
                        delete_model(db=db, storage=storage, model=model)
                    else:
                        raise HTTPException(
                            status_code=409,
                            detail=f"Model <{schema.model_name}:{schema.model_version}> already exists",
                        )

                # store the model in the shared repository, updating the model URI
                schema.model_uri = str(storage.save(schema, origin=tmp_repository))
                schema.source = schema.source or models_origin
                # store the model in the database
                model = Model(**schema.model_dump())
                db.add(model)
                db.commit()
                db.refresh(model)
                models.append(model)

        return models
    except (AssertionError, ValueError) as e:
        raise HTTPException(status_code=422, detail=f"Invalid source: {e}")


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
        updated_version = updates.version or model.model_version
        LOG.debug(f"Updating model {model.model_name}:{model.model_version} to {updated_name}:{updated_version}")
        # check if the model exists
        assert (
            get_model(db=db, storage=storage, model_name=updated_name, model_version=updated_version) is None
        ), f"Model <{updated_name}:{updated_version}> already exists"
        # update the model
        model.model_name = updated_name
        model.model_version = updated_version
        model.source = updates.source or model.source
        # update the model in the storage
        model.model_uri = str(storage.update(model, current_uri=Path(model.model_uri)))
        model.updated_at = timezone_aware_now()
        db.commit()
        db.refresh(model)
        return model
    except AssertionError as e:
        raise HTTPException(status_code=409, detail=f"Cannot update model: {e}")
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Cannot update model: {e}")


def delete_model(db: Session, storage: ModelStorage, model: Model) -> None:
    """
    Deletes a model given the name and the version.

    Args:
        storage (ModelStorage): The storage implementation to use.
        model (ModelSchema): The model to delete.

    Raises:
        HTTPException: If the model could not be deleted.

    Returns:
        None

    """
    try:
        LOG.debug(f"Deleting model {model.model_name}:{model.model_version}")
        storage.delete(model)
        db.delete(model)
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Cannot delete model: {e}")
    finally:
        db.commit()
