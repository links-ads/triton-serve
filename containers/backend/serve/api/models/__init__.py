import logging
from typing import List, Optional
from serve.api.models.dto import Model, ModelCreateSchema

from fastapi import APIRouter, Depends, Form, HTTPException, UploadFile, File
from serve.api.models import domain

router = APIRouter()


@router.get("/models", response_model=List[Model], status_code=200, tags=["models"])
def get_models(query_name: Optional[str] = None,
                  version: Optional[int] = None):
    """Retrieves a list of models, allowing to filter by query_name and versions.

    :param query_name: model query_name as specified by the user, defaults to None
    :type query_name: Optional[str], optional
    :param version: version of the model
    :type version: Optional[int], optional
    :return: list of models
    :rtype: List[Model]
    """
    models = domain.get_models(query_name=query_name, version=version)
    return models

@router.get("/models/{name}/{version}", response_model=Model, status_code=200, tags=["models"])
def get_model(name: str, version: int):
    """Retrieves the model corresponding to the provided name and version, if present.

    :param name: model name
    :type name: str
    :param version: model version
    :type version: int
    :return: corresponding model entity
    :rtype: Model
    """
    model = domain.get_model(name, version)
    if model is None:
        raise HTTPException(status_code=404, detail=f"Model with name={name} and version={version} does not exist")
    return model

@router.post("/models", status_code=201, tags=["models"])
async def post_model(model: ModelCreateSchema = Depends(), 
                     zip_file: UploadFile = File(...)
                     ):
    """Uploads a model to the model repository.

    :param model: model name and version
    :type model: ModelCreateSchema
    :param zip_file: zip file containing the model
    :type zip_file: UploadFile
    :return: corresponding model entity
    :rtype: Model
    """
    zip_file_bytes = await zip_file.read()

    model = domain.post_model(model.name, model.version, zip_file_bytes)
    return model