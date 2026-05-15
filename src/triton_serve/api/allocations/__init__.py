from typing import Any

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from triton_serve.api.allocations import domain
from triton_serve.database.schema import MachineAllocationSchema, ServiceAllocationSchema
from triton_serve.extensions import get_db
from triton_serve.security import require_elevated

router = APIRouter()


@router.get(
    "/resources",
    status_code=200,
    tags=["allocations"],
    response_model=MachineAllocationSchema,
)
def get_resources(
    db: Session = Depends(get_db),
    _: Any = Depends(require_elevated),
):
    """
    Returns the current resource allocation overview for the machine.

    Includes CPU, RAM, and per-GPU allocation broken down into:
    - `allocated`: reserved by all non-deleted services
    - `in_use`: consumed by services in STARTING or ACTIVE state

    **Returns:**
    - `MachineAllocationSchema`: Machine-wide resource snapshot.
    """
    return domain.get_resource_overview(db=db)


@router.get(
    "/resources/{service_id}",
    status_code=200,
    tags=["allocations"],
    response_model=ServiceAllocationSchema,
)
def get_service_resources(
    service_id: int,
    db: Session = Depends(get_db),
    _: Any = Depends(require_elevated),
):
    """
    Returns the resource allocation for a specific service.

    **Arguments:**
    - `service_id` (`int`): The id of the service.

    **Returns:**
    - `ServiceAllocationSchema`: CPU, RAM, and GPU allocation for the service.
    """
    return domain.get_service_allocation(db=db, service_id=service_id)
