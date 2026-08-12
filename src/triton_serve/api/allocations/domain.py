from fastapi import HTTPException
from sqlalchemy.orm import Session

from triton_serve.database.model import Machine, RuntimeStatus, Service
from triton_serve.database.schema import (
    DeviceAllocationSummarySchema,
    DeviceAllocationViewSchema,
    DeviceServiceSchema,
    MachineAllocationSchema,
    ResourceUsageSchema,
    ServiceAllocationSchema,
    ServiceDeviceAllocationSchema,
)

# a service occupies its allocated resources whenever the reconciler intends a container up
_IN_USE = {RuntimeStatus.READY, RuntimeStatus.WARMING, RuntimeStatus.RECOVERING}


def get_resource_overview(db: Session) -> MachineAllocationSchema:
    machine = db.query(Machine).first()
    if machine is None:
        raise HTTPException(status_code=404, detail="No machine registered")

    all_services = db.query(Service).filter(Service.deleted_at.is_(None)).all()

    cpu_allocated = sum(s.resources.cpu_count for s in all_services if s.resources)
    cpu_in_use = sum(s.resources.cpu_count for s in all_services if s.resources and s.runtime_status in _IN_USE)
    mem_allocated = sum(s.resources.mem_size for s in all_services if s.resources)
    mem_in_use = sum(s.resources.mem_size for s in all_services if s.resources and s.runtime_status in _IN_USE)

    devices = []
    for device in machine.devices:
        active_allocs = [a for a in device.allocations if a.service.deleted_at is None]
        allocated_pct = sum(a.allocation_percentage for a in active_allocs)
        in_use_pct = sum(a.allocation_percentage for a in active_allocs if a.service.runtime_status in _IN_USE)

        devices.append(
            DeviceAllocationViewSchema(
                uuid=device.uuid,
                name=device.name,
                index=device.index,
                memory=device.memory,
                allocation=DeviceAllocationSummarySchema(allocated_pct=allocated_pct, in_use_pct=in_use_pct),
                services=[
                    DeviceServiceSchema(
                        service_id=a.service_id,
                        service_name=a.service.service_name,
                        runtime_status=a.service.runtime_status,
                        allocation_percentage=a.allocation_percentage,
                    )
                    for a in active_allocs
                ],
            )
        )

    return MachineAllocationSchema(
        host_name=machine.host_name,
        num_cpus=machine.num_cpus,
        total_memory=machine.total_memory,
        cpu=ResourceUsageSchema(allocated=cpu_allocated, in_use=cpu_in_use),
        memory=ResourceUsageSchema(allocated=mem_allocated, in_use=mem_in_use),
        devices=devices,
    )


def get_service_allocation(db: Session, service_id: int) -> ServiceAllocationSchema:
    service = db.get(Service, ident=service_id)
    if service is None or service.deleted_at is not None:
        raise HTTPException(status_code=404, detail=f"Service with id {service_id} does not exist")

    res = service.resources
    return ServiceAllocationSchema(
        service_id=service.service_id,
        service_name=service.service_name,
        runtime_status=service.runtime_status,
        cpu_count=res.cpu_count,
        mem_size=res.mem_size,
        devices=[
            ServiceDeviceAllocationSchema(
                uuid=alloc.device.uuid,
                name=alloc.device.name,
                index=alloc.device.index,
                memory=alloc.device.memory,
                allocation_percentage=alloc.allocation_percentage,
            )
            for alloc in service.device_allocations
        ],
    )
