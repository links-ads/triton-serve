from pathlib import Path

from fastapi import HTTPException, UploadFile

from triton_serve.api.schema import ModelSchema
from triton_serve.storage.base import ModelStorage


def list_models(repository_path: Path, model_name: str | None = None, version: int | None = None) -> list[ModelSchema]:
    """
    Retrieves a list of models filtered by the given parameters, if provided.

    Args:
        repository_path (Path): The path to the directory containing the models.
        model_name (Optional[str], optional): The name of the model to filter. Defaults to None.
        version (Optional[int], optional): The version of the model to filter. Defaults to None.

    Returns:
        List[ModelSchema]: A list of ModelSchema instances representing the filtered models.
    """
    models = []

    if not repository_path.is_dir():
        return models  # Return an empty list if the models_path is not a directory
    model_dirs = repository_path.glob("*")

    for model_dir in model_dirs:
        if model_dir.is_dir():
            if model_name and model_name != model_dir.name:
                continue  # Skip if name doesn't match the model directory name

            versions = []
            if version:  # If version is specified, only check that specific version
                version_path = model_dir / str(version)
                if version_path.is_dir():
                    versions.append(version)
            else:  # If version is not specified, check all versions
                version_dirs = model_dir.glob("*")
                for version_dir in version_dirs:
                    if version_dir.is_dir():
                        try:
                            versions.append(int(version_dir.name))
                        except ValueError:
                            pass  # Skip non-integer version folders

            for ver in versions:
                models.append(ModelSchema(name=model_dir.name, version=ver))

    return models

def get_model(repository_path: Path, name: str, version: int) -> ModelSchema | None:
    """
    Retrieves a model given the name and version.

    Args:
        repository_path (Path): The path to the repository directory containing the models.
        name (str): The name of the model to retrieve.
        version (int): The version of the model to retrieve.

    Returns:
        ModelSchema: Returns the requested ModelSchema instance if found, or None otherwise.
    """
    model_path = Path(repository_path, name, str(version), "model.onnx")
    if not model_path.exists():
        return None

    model = ModelSchema(name=name, version=version)
    return model


def create_model(
    name: str,
    version: int,
    package: UploadFile,
    storage: ModelStorage,
) -> ModelSchema | None:
    """Uploads a model given the name and the version.

    :param name: filename of the model
    :type name: str
    :param version: model version
    :type version: int
    :param version: zip file containing the model.onnx and the pbtxt file (optional)
    :type version: str
    :return: a model entity, if present. Return None if model is not found
    :rtype: Optional[Model]
    """
    model = ModelSchema(name=name, version=version)
    try:
        storage.save(ModelSchema(name=name, version=version), package)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Invalid file uploaded: {e}")
    return model
