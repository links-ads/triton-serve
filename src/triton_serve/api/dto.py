from pydantic import BaseModel, field_validator


class ModelInfo(BaseModel):
    name: str
    version: int


class ModelUpdateBody(BaseModel):
    name: str | None = None
    version: int | None = None
    source: str | None = None


class ServiceCreateBody(BaseModel):
    name: str
    models: list[ModelInfo]
    gpus: int = 0
    docker_image: str | None = None
    environment: dict[str, str] | None = None

    @field_validator("name")
    @classmethod
    def validate_name(cls, v):
        if not v:
            raise ValueError("Service name cannot be empty")
        return v

    @field_validator("models")
    @classmethod
    def validate_models(cls, v):
        if not v:
            raise ValueError("Service must have at least one model")
        return v

    @field_validator("gpus")
    @classmethod
    def validate_gpus(cls, v):
        if v is None:
            raise ValueError("Number of GPUs cannot be empty")
        if not isinstance(v, int) or v < 0:
            raise ValueError("Number of GPUs needs to be a non-negative integer")
        return v
