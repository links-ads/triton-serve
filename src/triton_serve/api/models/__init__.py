from typing import Any

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.orm import Session

from triton_serve.api.dto import ModelUpdateBody
from triton_serve.api.models import domain
from triton_serve.config import AppSettings, get_settings, get_storage
from triton_serve.database.schema import ModelSchema
from triton_serve.extensions import get_db
from triton_serve.security import require_admin, require_elevated
from triton_serve.storage import ModelStorage
from triton_serve.storage.sources import ArchiveModelSource, RepositoryModelSource

router = APIRouter(prefix="/models")


@router.get(
    "",
    response_model=list[ModelSchema],
    status_code=200,
    tags=["models"],
)
def get_models(
    model_name: str | None = None,
    deleted: bool = False,
    db: Session = Depends(get_db),
    _: Any = Depends(require_elevated),
):
    """
    Retrieves a list of models, allowing filtering by `name` and `versions`.

    **Arguments:**
    - `model_name` (`Optional[str]`, optional): Model `name` as specified by the user. Defaults to `None`.
    - `deleted` (`Optional[bool]`, optional): Whether to include deleted models. Defaults to `False`.

    **Returns:**
    - `List[ModelSchema]`: A list of models.
    """
    if model_name == "":
        raise HTTPException(status_code=422, detail="Model name cannot be empty")
    models = domain.get_all_models(db=db, model_name=model_name, deleted=deleted)
    return models


@router.get(
    "/{model_name}",
    response_model=ModelSchema,
    status_code=200,
    tags=["models"],
)
def get_model(
    model_name: str,
    db: Session = Depends(get_db),
    _: Any = Depends(require_elevated),
):
    """
    Retrieves a specific model by id, if present.

    **Arguments:**
    - `model_name` (`str`): The name of the model.

    **Returns:**
    - `ModelSchema`: The requested model.
    """
    model = domain.get_single_model(db=db, model_name=model_name)
    if model is None:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' does not exist")
    return model


@router.post(
    "",
    status_code=201,
    response_model=list[ModelSchema],
    tags=["models"],
    operation_id="ModelsFromArchive",
)
def create_models_from_archive(
    package: UploadFile = File(...),
    update: bool = Form(False),
    db: Session = Depends(get_db),
    storage: ModelStorage = Depends(get_storage),
    settings: AppSettings = Depends(get_settings),
    _: Any = Depends(require_elevated),
):
    """
    Creates a new set of models, based on the provided compressed archive.

    **Arguments:**
    - `package` (`UploadFile`): Model package.
    - `update` (`bool`, optional): Whether to update existing models. Defaults to `False`.

    **Returns:**
    - `list[ModelSchema]`: The stored models.
    """
    source = ArchiveModelSource(package, target_dir=settings.repository_dirname)
    stored_models = domain.create_models_from_source(
        source=source,
        storage=storage,
        db=db,
        update=update,
    )
    return stored_models


@router.post(
    "/repository",
    response_model=list[ModelSchema],
    status_code=201,
    tags=["models"],
)
def create_models_from_repository(
    repository_url: str,
    update: bool = False,
    db: Session = Depends(get_db),
    storage: ModelStorage = Depends(get_storage),
    settings: AppSettings = Depends(get_settings),
    _: Any = Depends(require_elevated),
):
    """
    Creates models from a repository.

    **Arguments:**
    - `repository_url` (`URL`): The URL of the repository to clone, in SSH format.
    - `update` (`bool`, optional): Whether to update existing models. Defaults to `False`.

    **Returns:**
    - `List[ModelSchema]`: The created models.
    """
    try:
        source = RepositoryModelSource(repository_url, target_dir=settings.repository_dirname)
        stored_models = domain.create_models_from_source(
            source=source,
            storage=storage,
            db=db,
            update=update,
        )
        return stored_models
    except AssertionError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.put(
    "/{model_name}",
    response_model=ModelSchema,
    tags=["models"],
)
def rename_model(
    model_name: str,
    model_info: ModelUpdateBody,
    db: Session = Depends(get_db),
    storage: ModelStorage = Depends(get_storage),
    _: Any = Depends(require_elevated),
):
    """
    Updates a model by name and version.

    **Arguments:**
    - `model_name` (`str`): The name of the model.
    - `model_info` (`ModelUpdateBody`): The updates to apply to the model.

    **Returns:**
    - `ModelSchema`: The updated model.
    """
    model = domain.get_single_model(db=db, model_name=model_name)
    if model is None:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' does not exist")
    domain.edit_model_info(
        db=db,
        storage=storage,
        model=model,
        updates=model_info,
    )
    return model


@router.delete(
    "/{model_name}",
    status_code=200,
    tags=["models"],
)
def delete_model(
    model_name: str,
    model_version: int | None = None,
    db: Session = Depends(get_db),
    storage: ModelStorage = Depends(get_storage),
    _: Any = Depends(require_admin),
):
    """
    Deletes a model by name and version.

    **Arguments:**
    - `model_name` (`str`): The name of the model, to delete everything.
    - `model_version` (`int`): The version of the model, to delete only a specific version.

    **Returns:**
    - `200` if the model was deleted successfully.
    """
    model = domain.get_single_model(db=db, model_name=model_name)
    if model is None:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' does not exist")
    # do not allow deletion of models with associated active services
    for service in model.services:
        if service.deleted_at is None:
            raise HTTPException(
                status_code=409,
                detail=f"Model '{model_name}' has associated services. Please delete the services first.",
            )
    if model_version is not None:
        if not any(version.version_id == model_version for version in model.versions):
            raise HTTPException(status_code=404, detail=f"Model version '{model_name}:{model_version}' does not exist")
    return domain.delete_model(db=db, storage=storage, model=model, version_number=model_version)
