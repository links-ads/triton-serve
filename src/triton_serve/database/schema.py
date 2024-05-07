from datetime import datetime, timezone

from pydantic import BaseModel, ConfigDict

from triton_serve.database.model import ModelType, ServiceStatus


def timezone_aware_now():
    return datetime.now(tz=timezone.utc)


class MachineBaseSchema(BaseModel):
    host_name: str
    num_cpus: int
    total_memory: int


class MachineCreateSchema(MachineBaseSchema):
    pass


class MachineSchema(MachineBaseSchema):
    model_config = ConfigDict(from_attributes=True)
    host_id: int


class ModelBaseSchema(BaseModel):
    # remove constraints on model_ variables
    model_config = ConfigDict(protected_namespaces=())
    model_name: str
    model_version: int
    model_type: ModelType
    model_uri: str
    created_at: datetime = timezone_aware_now()
    updated_at: datetime = timezone_aware_now()
    source: str | None
    dependencies: list[str] | None = None


class ModelCreateSchema(ModelBaseSchema):
    pass


class ModelSchema(ModelBaseSchema):
    model_config = ConfigDict(from_attributes=True)


class DeviceBaseSchema(BaseModel):
    uuid: str
    name: str
    memory: int
    index: int


class DeviceCreateSchema(DeviceBaseSchema):
    pass


class DeviceSchema(DeviceBaseSchema):
    model_config = ConfigDict(from_attributes=True)
    host_id: int


class ServiceBaseSchema(BaseModel):
    service_name: str
    service_image: str
    container_id: str | None = None
    container_status: ServiceStatus = ServiceStatus.STARTING
    created_at: datetime
    deleted_at: datetime | None = None
    cpu_count: int
    mem_size: int
    shm_size: int
    models: list[ModelSchema] = []
    devices: list[DeviceSchema] = []


class ServiceCreateSchema(ServiceBaseSchema):
    pass


class ServiceSchema(ServiceBaseSchema):
    model_config = ConfigDict(from_attributes=True)
    service_id: int
