from fastapi import Form
from pydantic import BaseModel, ValidationError, validator

class Model(BaseModel):
    name: str
    version: int


    
    def __init__(self, name: str, version: str):
        super().__init__(name=name, version=version)




    def __repr__(self):
        return f"<Model {self.name} v{self.version}>"


class ModelCreateSchema(BaseModel):
    name: str
    version: int

    def __init__(self, name: str = Form(...), version: str = Form(1)):
        super().__init__(name=name, version=version)

    @validator("name")
    def validate_name(cls, v):
        if v is None:
            raise ValueError("Model name cannot be empty")
        return v
    
    @validator("version")
    def validate_version(cls, v):
        if v is None:
            raise ValueError("Model version cannot be empty")
        if not isinstance(v, int) or v < 0:
            raise ValueError("Model version needs to be a positive integer number")
        return v
    



