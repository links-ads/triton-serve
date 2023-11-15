from fastapi import Form
from pydantic import BaseModel, ValidationError, validator
from typing import List


class ServiceCreateSchema(BaseModel):
    name: str
    models: List[str]