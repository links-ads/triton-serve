from pydantic import BaseModel


# MACHINES
class MachineBase(BaseModel):
    host_name: str
    num_cpus: int
    total_memory: int


class MachineCreate(MachineBase):
    pass


class Machine(MachineBase):
    host_id: int

    class Config:
        orm_mode = True


# DEVICES
class DeviceBase(BaseModel):
    uuid: str
    name: str
    memory: int
    index: int


class DeviceCreate(DeviceBase):
    pass


class Device(DeviceBase):
    host_id: int

    class Config:
        orm_mode = True


# SERVICES
class ServiceBase(BaseModel):
    service_name: str
    models: list[str]
    created_at: int
    assigned_device: str | None


class ServiceCreate(ServiceBase):
    pass


class Service(ServiceBase):
    service_id: str

    class Config:
        orm_mode = True
