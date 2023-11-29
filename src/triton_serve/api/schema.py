from fastapi import Form
from pydantic import BaseModel, field_validator


class ModelSchema(BaseModel):
    name: str
    version: int


class ModelCreateSchema(ModelSchema):
    def __init__(self, name: str = Form(...), version: int = Form(1)):
        super().__init__(name=name, version=version)

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


class ServiceCreateSchema(BaseModel):
    name: str
    models: list[str]

    @field_validator("name")
    def validate_name(cls, v):
        if not v:
            raise ValueError("Service name cannot be empty")
        return v

    @field_validator("models")
    def validate_models(cls, v):
        if v is None:
            raise ValueError("Model cannot be empty")
        return v


class ServiceSchema(BaseModel):
    name: str
    models: list[ModelSchema]
