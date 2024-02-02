from fastapi import HTTPException, UploadFile
from sqlalchemy.orm import Session

from triton_serve.database.model import Model
from triton_serve.storage.base import ModelStorage


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


def create_model(
    name: str,
    version: int,
    description: str | None,
    package: UploadFile,
    storage: ModelStorage,
    db: Session,
) -> Model | None:
    """Uploads a model given the name and the version.

    Args:
        name (str): The name of the model to create.
        version (int): The version of the model to create.
        description (str, optional): The description of the model to create. Defaults to None.
        package (UploadFile): The package containing the model.
        storage (ModelStorage): The storage implementation to use.
        db (Session): The database session.

    Returns:
        ModelSchema: Returns the created ModelSchema instance.

    Raises:
        HTTPException: If the model already exists.
        HTTPException: If the file is invalid.
    """
    model = Model(
        model_name=name,
        model_version=version,
        description=description,
    )
    try:
        model.model_uri = str(storage.save(model, package))
        db.add(model)
        db.commit()
        db.refresh(model)
    except AssertionError as e:
        raise HTTPException(status_code=422, detail=f"Malformed archive: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cannot create model: {e}")
    return model


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
        storage.delete(model)
        db.delete(model)
        db.commit()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Can't delete model: {e}")
