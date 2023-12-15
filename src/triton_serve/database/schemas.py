from pydantic import BaseModel, ConfigDict


# MACHINES
class MachineBase(BaseModel):
    host_name: str
    num_cpus: int
    total_memory: int


class MachineCreate(MachineBase):
    pass


class Machine(MachineBase):
    model_config = ConfigDict(from_attributes=True)
    host_id: int


# DEVICES
class DeviceBase(BaseModel):
    uuid: str
    name: str
    memory: int
    index: int


class DeviceCreate(DeviceBase):
    pass


class Device(DeviceBase):
    model_config = ConfigDict(from_attributes=True)
    host_id: int


# SERVICES
class ServiceBase(BaseModel):
    service_name: str
    models: list[str]
    created_at: int
    assigned_device: str | None


class ServiceCreate(ServiceBase):
    pass


class Service(ServiceBase):
    model_config = ConfigDict(from_attributes=True)
    service_id: str
