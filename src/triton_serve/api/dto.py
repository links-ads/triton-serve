from fastapi import Form
from pydantic import BaseModel, field_validator


class ModelCreateBody(BaseModel):
    name: str
    version: int
    description: str | None

    def __init__(self, name: str = Form(...), version: int = Form(1), description: str = Form(None)):
        super().__init__(name=name, version=version, description=description)

    @field_validator("name")
    def validate_name(cls, v):
        if not v:
            raise ValueError("Model name cannot be empty")
        return v

    @field_validator("version")
    def validate_version(cls, v):
        if v is None:
            raise ValueError("Model version cannot be empty")
        if not isinstance(v, int) or v < 1:
            raise ValueError("Model version needs to be a positive integer >= 1")
        return v


class ServiceCreateModel(BaseModel):
    name: str
    version: int


class ServiceCreateBody(BaseModel):
    name: str
    models: list[ServiceCreateModel]
    gpu: bool = False
    docker_image: str | None = None

    @field_validator("name")
    def validate_name(cls, v):
        if not v:
            raise ValueError("Service name cannot be empty")
        return v


class GPUDevice(BaseModel):
    uuid: str
    name: str
    memory: int
    index: int


class MachineSchema(BaseModel):
    host_name: str
    num_cpus: int
    total_memory: int
    devices: list[GPUDevice] = []
