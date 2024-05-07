from pydantic import BaseModel, Field, field_validator


class ModelInfo(BaseModel):
    name: str
    version: int


class ModelUpdateBody(BaseModel):
    name: str | None = None
    version: int | None = None
    source: str | None = None


class ServiceResources(BaseModel):
    gpus: int = Field(ge=0, default=0, description="Number of GPUs")
    shm_size: int = Field(gt=0, le=65536, default=256, description="Shared memory size in MB")
    mem_size: int = Field(gt=0, le=65536, default=4096, description="Memory size in MB")
    cpu_count: int = Field(gt=0, default=2, description="Number of CPUs")

    @field_validator("shm_size", "mem_size", mode="before")
    @classmethod
    def validate_units(cls, value: int | str) -> int:
        if isinstance(value, str):
            value = value.upper()
            if value[-1] == "M":
                return int(value[:-1])
            if value[-1] == "G":
                return int(value[:-1]) * 1024
            raise ValueError("Invalid unit")
        return value


class ServiceCreateBody(BaseModel):
    name: str
    models: list[ModelInfo]
    docker_image: str | None = None
    environment: dict[str, str] | None = None
    resources: ServiceResources = ServiceResources()

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
