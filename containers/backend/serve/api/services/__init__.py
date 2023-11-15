import logging
from typing import List, Optional
from serve.api.services.dto import ServiceCreateSchema

from fastapi import APIRouter, Depends, Form, HTTPException, UploadFile, File
from serve.api.services import domain

router = APIRouter()


@router.post("/services", status_code=201, tags=["services"])
async def post_model(service: ServiceCreateSchema
                     ):

    service = domain.post_service(service.name, service.models)
    return service