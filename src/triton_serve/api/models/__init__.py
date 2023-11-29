from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from triton_serve.api.models import domain
from triton_serve.api.schema import ModelCreateSchema, ModelSchema
from triton_serve.config import AppSettings, get_settings, get_storage
from triton_serve.storage import ModelStorage

router = APIRouter()


@router.get("/models", response_model=list[ModelSchema], status_code=200, tags=["models"])
def get_models(
    model_name: str | None = None,
    version: int | None = None,
    settings: AppSettings = Depends(get_settings),
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
    models_path = settings.repository_path
    models = domain.list_models(models_path, model_name=model_name, version=version)
    return models


@router.get("/models/{name}/{version}", response_model=ModelSchema, status_code=200, tags=["models"])
def get_model(
    name: str,
    version: int,
    settings: AppSettings = Depends(get_settings),
):
    """
    Retrieves a specific model by name and version, if present.

    **Arguments:**
    - `name` (str): The name of the model.
    - `version` (int): The version of the model.

    **Returns:**
    - `Model`: The requested model.
    """
    repository_path = settings.repository_path
    model = domain.get_model(repository_path, name, version)
    if model is None:
        raise HTTPException(status_code=404, detail=f"Model with name={name} and version={version} does not exist")
    return model


@router.post(
    "/models",
    status_code=201,
    generate_unique_id_function=lambda _: "ModelCreateSchema",
    tags=["models"],
)
async def create_model(
    model: ModelCreateSchema = Depends(),
    package: UploadFile = File(...),
    settings: AppSettings = Depends(get_settings),
    storage: ModelStorage = Depends(get_storage),
):
    """
    Creates a new model.

    **Arguments:**
    - `name` (`str`): Model name, this should be unique across models.
    - `version` (`int`): Model version, this should be unique across versions of the same model.
    - `package` (`UploadFile`): Model package.

    **Returns:**
    - `Model`: The created model.
    """
    repository_path = settings.repository_path
    if domain.get_model(repository_path, model.name, model.version) is not None:
        raise HTTPException(status_code=409, detail=f"Model {model} already exists")
    model = domain.create_model(
        name=model.name,
        version=model.version,
        package=package,
        storage=storage,
    )
    return model


@router.delete("/models/{name}/{version}", status_code=204, tags=["models"])
def delete_model(
    name: str,
    version: int,
    settings: AppSettings = Depends(get_settings),
    storage: ModelStorage = Depends(get_storage),
):
    """
    Deletes a model by name and version.

    **Arguments:**
    - `name` (`str`): The name of the model.
    - `version` (`int`): The version of the model.

    **Returns:**
    - `None`
    """

    repository_path = settings.repository_path
    model = domain.get_model(repository_path, name, version)
    if model is None:
        raise HTTPException(status_code=404, detail=f"Model with name={name} and version={version} does not exist")
    domain.delete_model(storage, model)
