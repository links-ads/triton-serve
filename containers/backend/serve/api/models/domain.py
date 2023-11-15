import logging
import shutil
from typing import List, Optional

from fastapi import HTTPException
from serve.api.models.dto import Model
from serve.common import utils


from pathlib import Path
from serve.common.constants import BASE_PATH, LOCAL_MODEL_REPOSITORY


def get_models(query_name: Optional[str] = None, version: Optional[int] = None) -> List[Model]:
    """Retrieves a list of models, filtered by the given parameters, if present.

    :param query_name: name of the model, to filter some models that contain the same string, defaults to None
    :type query_name: Optional[str], optional
    :param version: model version, defaults to None
    :type version: Optional[int], optional
    :return: a list of model entities or an empty list, if none have been found
    :rtype: List[Model]
    """

    models = []

    # add the models folder to the path
    models_path = LOCAL_MODEL_REPOSITORY
    
    for model_path in models_path.iterdir():
        if model_path.is_dir():
            model_name = model_path.parts[-1]
            if query_name is not None and not (query_name in model_name):
                continue

            versions_list = list(model_path.glob("**/*.onnx"))

            for version_path in versions_list:
                if version_path.parts[-1] != "model.onnx":
                    continue
                version = version_path.parts[-2]
                if version is not None and version != str(version):
                    continue
                model = Model(name=model_name, version=version)
                models.append(model)

            
    return models


def get_model(name: str, version: int) -> Optional[Model]:
    """Retrieves a model given the name and the version.

    :param name: filename of the model
    :type query_name: str
    :param version: model version
    :type version: int
    :return: a model entity, if present. Return None if model is not found
    :rtype: Optional[Model]
    """
    models_path = LOCAL_MODEL_REPOSITORY

    model_path = Path(models_path, name, str(version), "model.onnx")
    if not model_path.exists():
        return None

    model = Model(name=name, version=version)
    return model


def post_model(name: str, version: int, model_file: bytes) -> Optional[Model]:
    """Uploads a model given the name and the version.

    :param name: filename of the model
    :type query_name: str
    :param version: model version
    :type version: int
    :param version: zip file containing the model.onnx and the pbtxt file (optional)
    :type version: str
    :return: a model entity, if present. Return None if model is not found
    :rtype: Optional[Model]
    """
    models_path = LOCAL_MODEL_REPOSITORY
    temp_folder = Path(BASE_PATH, "containers","backend", "serve", "common", "temp", name + str(version))

    model_path = Path(models_path, name, str(version), "model.onnx")
    if model_path.exists():
        return HTTPException(status_code=409, detail=f"Model with name={name} and version={version} already exists")

    paths = utils.unzip_model(model_file, name, version)

    if paths == "Internal Server Error":
        shutil.rmtree(temp_folder)
        return HTTPException(status_code=500, detail=f"Internal Server Error")
    elif paths == "Request conflict":
        return HTTPException(status_code=400, detail=f"Something went wrong, retry in a few seconds")
    elif paths == "File number error":
        return HTTPException(status_code=422, detail=f"Wrong number of files in the zip file. Make sure to upload only the model.onnx file or both the model.onnx and the config.pbtxt file")
    elif paths == "No model.onnx":
        return HTTPException(status_code=422, detail=f"model.onnx file not found in the zip file")
    elif paths == "No config.pbtxt":
        return HTTPException(status_code=422, detail=f"config.pbtxt file not found in the zip file")
    elif paths == "zip file error":
        return HTTPException(status_code=422, detail=f"Error while unzipping the zip file, make sure it's not corrupted")
    

    # create teh destination folder
    dest_folder = Path(models_path, name, str(version))
    dest_folder.mkdir(parents=True, exist_ok=True)


    
    # move the files to the model repository
    for path in paths:
        dest = Path(models_path, name, str(version), path.name)
        dest_path = path.replace(dest)
    
    
    # remove the temp folder and its content
    shutil.rmtree(temp_folder)

    model = Model(name=name, version=version)
    return model

