from datetime import datetime, timezone

from pydantic import BaseModel, ConfigDict, Field

from triton_serve.database.model import KeyType, ModelType, ServiceStatus


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


class DeviceBaseSchema(BaseModel):
    uuid: str
    name: str
    memory: int
    index: int


class DeviceCreateSchema(DeviceBaseSchema):
    host_id: int | None = None


class DeviceSchema(DeviceBaseSchema):
    model_config = ConfigDict(from_attributes=True)
    host_id: int


class ModelVersionBaseSchema(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    version_id: int
    model_uri: str


class ModelVersionCreateSchema(ModelVersionBaseSchema):
    model_id: int | None = None


class ModelVersionSchema(ModelVersionBaseSchema):
    model_config = ConfigDict(
        from_attributes=True,
        protected_namespaces=(),
    )
    model_id: int


class ModelBaseSchema(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    model_name: str
    model_type: ModelType
    created_at: datetime = Field(default_factory=timezone_aware_now)
    updated_at: datetime = Field(default_factory=timezone_aware_now)
    source: str | None = None
    dependencies: list | None = Field(default_factory=list)
    version_policy: dict | None = None
    versions: list[ModelVersionBaseSchema] = Field(default_factory=list)


class ModelCreateSchema(ModelBaseSchema):
    pass


class ModelSchema(ModelBaseSchema):
    model_config = ConfigDict(from_attributes=True)
    model_id: int


class ServiceResourcesSchema(BaseModel):
    cpu_count: int = Field(gt=0)
    shm_size: int = Field(gt=0)
    mem_size: int = Field(gt=0)
    environment_variables: dict | None = None


class DeviceAllocationSchema(BaseModel):
    device_id: str
    allocation_percentage: float = Field(gt=0, le=100)


class ServiceBaseSchema(BaseModel):
    service_name: str
    service_image: str
    container_id: str | None = None
    container_status: ServiceStatus = ServiceStatus.STARTING
    created_at: datetime = Field(default_factory=timezone_aware_now)
    deleted_at: datetime | None = None
    inactivity_timeout: int = Field(default=3600, ge=0)
    priority: int = Field(default=0, ge=0)
    last_active_time: datetime | None = None
    resources: ServiceResourcesSchema
    models: list[ModelSchema] = Field(default_factory=list)
    device_allocations: list[DeviceAllocationSchema] = Field(default_factory=list)


class ServiceInfoSchema(BaseModel):
    service_id: int
    service_name: str
    container_id: str
    container_status: ServiceStatus


class ServiceCreateSchema(ServiceBaseSchema):
    pass


class ServiceSchema(ServiceBaseSchema):
    model_config = ConfigDict(from_attributes=True)
    service_id: int


class APIKeyBaseSchema(BaseModel):
    key_type: KeyType
    value: str
    project: str
    notes: str | None = None
    created_at: datetime = Field(default_factory=timezone_aware_now)
    expires_at: datetime | None = None
    services: list[ServiceInfoSchema] = []


class APIKeyCreateSchema(APIKeyBaseSchema):
    pass


class APIKeySchema(APIKeyBaseSchema):
    model_config = ConfigDict(from_attributes=True)
    key_id: int
