import logging
import re
import subprocess
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from packaging.specifiers import SpecifierSet
from packaging.version import Version

from triton_serve.database.model import ModelType
from triton_serve.database.schema import ModelCreateSchema, ModelVersionCreateSchema

# Regular expressions to extract specific fields
REGEX_FIELDS = {
    "name": re.compile(r'^name:\s*"([^"]+)"', re.MULTILINE),
    "platform": re.compile(r'^platform:\s*"([^"]+)"', re.MULTILINE),
}
LOG = logging.getLogger("uvicorn")


def parse_config(config_file: Path) -> dict[str, str]:
    """Validates the content of the given config file.

    Args:
        config_file (Path): path to the config file

    Returns:
        dict[str, str]: dictionary containing the extracted fields
    """
    result = {}
    with open(config_file) as f:
        content = f.read()
        for field, regex in REGEX_FIELDS.items():
            match = regex.search(content)
            result[field] = match.group(1) if match else None
    return result


def parse_version_policy(config_file: Path) -> dict:
    with open(config_file, "r") as f:
        content = f.read()

    policy = {}
    if "version_policy: { all: {}}" in content:
        policy["all"] = {}
    elif "version_policy: { latest:" in content:
        match = re.search(r"version_policy: { latest: { num_versions: (\d+)}}", content)
        if match:
            policy["latest"] = {"num_versions": int(match.group(1))}
    elif "version_policy: { specific:" in content:
        match = re.search(r"version_policy: { specific: { versions: \[([\d,\s]+)\]}}", content)
        if match:
            versions = [int(v.strip()) for v in match.group(1).split(",")]
            policy["specific"] = {"versions": versions}

    return policy


def parse_requirements(requirements_file: Path) -> list[str]:
    """Parse requirements.txt file into a list of requirements.

    Args:
        requirements_file (Path): path to the requirements file

    Returns:
        list[str]: list containing the requirements as strings.
    """
    dependencies = []
    if requirements_file.exists() and requirements_file.is_file():
        for line in requirements_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                dependencies.append(line)
    return dependencies


# the python a built image actually runs: the tritonserver base images ship 3.10
RUNTIME_PYTHON = Version("3.10")


@dataclass(frozen=True)
class BundleDependencies:
    pip: list[str] = field(default_factory=list)
    system: list[str] = field(default_factory=list)


def _export_locked(bundle_path: Path) -> list[str] | None:
    """Exports a committed uv.lock to fully pinned requirements, or None if there is no lock.

    A resolved closure is what makes the build hash honest: without it, an unpinned `numpy`
    resolves differently next month while the content hash still claims the image is unchanged.
    """
    if not (bundle_path / "uv.lock").is_file():
        return None
    result = subprocess.run(
        ["uv", "export", "--frozen", "--no-hashes", "--no-emit-project", "--format", "requirements-txt"],
        cwd=bundle_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"uv.lock is present but does not export: {result.stderr.strip()}"
    return [line.strip() for line in result.stdout.splitlines() if line.strip() and not line.startswith("#")]


def _parse_pyproject(manifest: Path) -> BundleDependencies:
    try:
        content = tomllib.loads(manifest.read_text())
    except tomllib.TOMLDecodeError as e:
        raise AssertionError(f"invalid pyproject.toml: {e}") from e

    project = content.get("project", {})
    if requires_python := project.get("requires-python"):
        assert RUNTIME_PYTHON in SpecifierSet(requires_python), (
            f"bundle requires-python {requires_python!r} excludes the runtime python {RUNTIME_PYTHON}"
        )

    serve = content.get("tool", {}).get("serve", {})
    return BundleDependencies(
        pip=_export_locked(manifest.parent) or list(project.get("dependencies", [])),
        system=list(serve.get("system-packages", [])),
    )


def parse_dependencies(bundle_path: Path) -> BundleDependencies:
    """Reads a bundle's dependency manifest.

    `pyproject.toml` is preferred and `requirements.txt` remains supported so existing bundles keep
    working untouched. Both normalize to the same shape here, so nothing downstream knows which
    format a bundle used.

    Args:
        bundle_path (Path): The root of the extracted bundle.

    Returns:
        BundleDependencies: The pip and system dependency lists, empty when there is no manifest.

    Raises:
        AssertionError: If the manifest is malformed, its lock does not export, or its
            requires-python excludes the runtime python.
    """
    if (manifest := bundle_path / "pyproject.toml").is_file():
        return _parse_pyproject(manifest)
    return BundleDependencies(pip=parse_requirements(bundle_path / "requirements.txt"))


def infer_model_type(model_name: str, files: list[Path]) -> ModelType:
    """Infers the model type from the given list of files.

    Args:
        files (list[Path]): list of files to analyze

    Returns:
        ModelType: the model type inferred from the files
    """
    extensions = {file.suffix for file in files if file.is_file()}
    match extensions:
        case exts if ".plan" in exts:
            return ModelType.TENSORRT
        case exts if ".onnx" in exts:
            return ModelType.ONNX
        case exts if ".pt" in exts:
            return ModelType.TORCHSCRIPT
        case exts if ".graphdef" in exts:
            return ModelType.TENSORFLOW
        case exts if any(file.is_dir() and file.name == "model.savedmodel" for file in files):
            return ModelType.TENSORFLOW
        case exts if ".xml" in exts and ".bin" in exts:
            return ModelType.OPENVINO
        case exts if ".py" in exts:
            return ModelType.PYTHON
        case exts if ".dali" in exts:
            return ModelType.DALI
        case _:
            raise AssertionError(f"{model_name}: unable to determine model type from files")


def validate_models(repository_path: Path) -> list[ModelCreateSchema]:
    """Validates the content of the given path to ensure it's compliant with
    a Triton model repository:
    - Each subdirectory must be a model
    - Each model must contain a config.pbtxt file
    - Each model must contain a version folder
    - Each version folder must contain at least one file, with different extensions
    - The model type is inferred from the files

    Args:
        repository_path (Path): path to the repository

    Returns:
        list[ModelCreateSchema]: list of validated models
    """
    models = []
    #  list all directories in the repository
    model_dirs = [d for d in repository_path.iterdir() if d.is_dir()]
    assert model_dirs, "Empty repository"

    dependencies = parse_dependencies(repository_path)

    for model_dir in model_dirs:
        model_name = model_dir.name
        config_file = model_dir / "config.pbtxt"
        assert config_file.exists() and config_file.is_file(), f"Missing or invalid config file in {model_name}"

        # check for version folders
        version_dirs = [d for d in model_dir.iterdir() if d.is_dir()]
        assert version_dirs, f"Empty model: {model_name}"

        # parse the config file, try to extract name and platform
        # if platform is not present, infer the model type from the files
        config = parse_config(config_file)
        version_policy = parse_version_policy(config_file)
        # check if the model name matches the one in the config file
        if config["name"] is not None:
            assert config["name"] == model_name, f"Model name mismatch in {model_name}"
        # check if the model has a platform
        if not (platform := config.get("platform")):
            version_files = list(version_dirs[0].iterdir())
            model_type = infer_model_type(model_name, version_files)
        else:
            model_type = ModelType(platform)
        LOG.debug(f"Model {model_name} is of type {model_type}")

        # create the model schema
        model = ModelCreateSchema(
            model_name=model_name,
            model_type=model_type,
            source=None,
            dependencies=dependencies.pip,
            system_dependencies=dependencies.system,
            version_policy=version_policy,
        )
        # add the versions and validate them
        for version_dir in version_dirs:
            assert version_dir.name.isdigit(), f"Invalid version in {model_name}:{version_dir.name}"
            files = list(version_dir.iterdir())
            assert files, f"Empty version in {model_name}:{version_dir.name}"
            model.versions.append(
                ModelVersionCreateSchema(
                    version_id=int(version_dir.name),
                    model_uri=str(version_dir),
                )
            )
        models.append(model)
    return models
