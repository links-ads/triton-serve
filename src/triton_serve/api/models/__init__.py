from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy.orm import Session

from triton_serve.api.dto import ModelCreateBody
from triton_serve.api.models import domain
from triton_serve.config import get_storage
from triton_serve.database.schema import ModelSchema
from triton_serve.extensions import get_db
from triton_serve.storage import ModelStorage

router = APIRouter()


@router.get("/models", response_model=list[ModelSchema], status_code=200, tags=["models"])
def get_models(
    model_name: str | None = None,
    version: int | None = None,
    db: Session = Depends(get_db),
    storage: ModelStorage = Depends(get_storage),
):
    """
    Retrieves a list of models, allowing filtering by `name` and `versions`.

    **Arguments:**
    - `name` (`Optional[str]`, optional): Model `name` as specified by the user. Defaults to `None`.
    - `version` (`Optional[int]`, optional): Version of the model. Defaults to `None`.

    **Returns:**
    - `List[Model]`: A list of models.
    """
    if model_name == "":
        raise HTTPException(status_code=422, detail="Model name cannot be empty")
    models = domain.list_models(
        db=db,
        storage=storage,
        model_name=model_name,
        version=version,
    )
    return models


@router.get("/models/{model_name}/{model_version}", response_model=ModelSchema, status_code=200, tags=["models"])
def get_model(
    model_name: str,
    model_version: int,
    db: Session = Depends(get_db),
    storage: ModelStorage = Depends(get_storage),
):
    """
    Retrieves a specific model by id, if present.

    **Arguments:**
    - `model_name` (`str`): The name of the model.
    - `model_version` (int): The version of the model.

    **Returns:**
    - `Model`: The requested model.
    """
    model = domain.get_model(
        db=db,
        model_name=model_name,
        model_version=model_version,
        storage=storage,
    )
    if model is None:
        raise HTTPException(status_code=404, detail=f"Model <{model_name}:{model_version}> does not exist")
    return model


@router.post(
    "/models",
    status_code=201,
    generate_unique_id_function=lambda _: "ModelCreate",
    tags=["models"],
)
def create_model(
    model_info: ModelCreateBody = Depends(),
    package: UploadFile = File(...),
    db: Session = Depends(get_db),
    storage: ModelStorage = Depends(get_storage),
):
    """
    Creates a new model.

    **Arguments:**
    - `name` (`str`): Model name, this should be unique across models.
    - `version` (`int`): Model version, this should be unique across versions of the same model.
    - `package` (`UploadFile`): Model package.
    - `description` (`Optional[str]`, optional): Model description. Defaults to `None`.

    **Returns:**
    - `Model`: The created model.
    """
    available_models = domain.list_models(
        db=db,
        storage=storage,
        model_name=model_info.name,
        version=model_info.version,
    )
    if available_models:
        raise HTTPException(status_code=409, detail=f"Model <{model_info}> already exists")
    model_info = domain.create_model(
        name=model_info.name,
        version=model_info.version,
        description=model_info.description,
        package=package,
        storage=storage,
        db=db,
    )
    return model_info


@router.delete("/models/{model_name}/{model_version}", status_code=204, tags=["models"])
def delete_model(
    model_name: str,
    model_version: int,
    db: Session = Depends(get_db),
    storage: ModelStorage = Depends(get_storage),
):
    """
    Deletes a model by name and version.

    **Arguments:**
    - `model_name` (`str`): The name of the model.
    - `model_version` (`int`): The version of the model.

    **Returns:**
    - `None`
    """
    model = domain.get_model(db=db, storage=storage, model_name=model_name, model_version=model_version)
    if model is None:
        raise HTTPException(status_code=404, detail=f"Model <{model_name}:{model_version}> does not exist")
    domain.delete_model(db=db, storage=storage, model=model)
