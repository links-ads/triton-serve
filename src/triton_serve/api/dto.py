from pydantic import BaseModel, Field, field_validator

from triton_serve.database.model import KeyType


class APIKeyCreateBody(BaseModel):
    project: str
    key_type: KeyType
    notes: str | None = None
    expiration_days: int = 365

    @field_validator("expiration_days")
    @classmethod
    def validate_expiration_days(cls, v):
        if v < 1:
            raise ValueError("Expiration days must be greater than 0")
        return v


class APIKeyUpdateBody(BaseModel):
    project: str | None = None
    notes: str | None = None

    @classmethod
    @field_validator("project")
    def validate_project(cls, v):
        if v is not None:
            if not v.strip():
                raise ValueError("Project name cannot be empty or just whitespace")
        return v


class ServiceKeyCreateBody(BaseModel):
    project: str
    notes: str | None = None
    expiration_days: int = 365

    @field_validator("expiration_days")
    @classmethod
    def validate_expiration_days(cls, v):
        if v < 1:
            raise ValueError("Expiration days must be greater than 0")
        return v


class ModelUpdateBody(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=255)
    source: str | None = None

    @classmethod
    @field_validator("name")
    def validate_name(cls, v):
        if v is not None:
            if not v.strip():
                raise ValueError("Name cannot be empty or just whitespace")
            if not v.isascii():
                raise ValueError("Name must contain only ASCII characters")
        return v


class ServiceCreateResources(BaseModel):
    gpus: float = Field(ge=0.0, default=0.0, description="Number of GPUs, float for fractional GPUs")
    shm_size: int = Field(gt=0, le=65536, default=256, description="Shared memory size in MB")
    mem_size: int = Field(gt=0, le=65536, default=4096, description="Memory size in MB")
    cpu_count: int = Field(gt=0, default=2, description="Number of CPUs")

    @classmethod
    @field_validator("shm_size", "mem_size", mode="before")
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
    models: list[str]
    docker_image: str | None = None
    environment: dict[str, str] = Field(default_factory=dict)
    timeout: int = 3600  # in seconds
    priority: int = 1  # higher number means higher priority
    resources: ServiceCreateResources = ServiceCreateResources()

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
