from datetime import datetime, timezone

from pydantic import BaseModel, ConfigDict, Field

from triton_serve.database.model import DesiredState, KeyType, ModelType, RuntimeStatus


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
    healthcheck: dict | None = None


class DeviceAllocationSchema(BaseModel):
    device_id: str
    allocation_percentage: float = Field(gt=0, le=100)


class ServiceBaseSchema(BaseModel):
    service_name: str
    service_image: str
    container_id: str | None = None
    created_at: datetime = Field(default_factory=timezone_aware_now)
    deleted_at: datetime | None = None
    inactivity_timeout: int = Field(default=3600, ge=0)
    priority: int = Field(default=0, ge=0)
    restart_attempts: int = 0
    last_attempt_at: datetime | None = None
    last_active_time: datetime | None = None
    resources: ServiceResourcesSchema
    models: list[ModelSchema] = Field(default_factory=list)
    device_allocations: list[DeviceAllocationSchema] = Field(default_factory=list)


class ServiceInfoSchema(BaseModel):
    service_id: int
    service_name: str
    container_id: str | None = None
    runtime_status: RuntimeStatus


class ServiceCreateSchema(ServiceBaseSchema):
    pass


class ServiceSchema(ServiceBaseSchema):
    model_config = ConfigDict(from_attributes=True)
    service_id: int
    desired_state: DesiredState
    runtime_status: RuntimeStatus


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


class QueueMessageDeleteResponseSchema(BaseModel):
    deleted_messages: int


class ResourceUsageSchema(BaseModel):
    allocated: int
    in_use: int


class DeviceServiceSchema(BaseModel):
    service_id: int
    service_name: str
    runtime_status: RuntimeStatus
    allocation_percentage: float


class DeviceAllocationSummarySchema(BaseModel):
    allocated_pct: float
    in_use_pct: float


class DeviceAllocationViewSchema(BaseModel):
    uuid: str
    name: str
    index: int
    memory: int
    allocation: DeviceAllocationSummarySchema
    services: list[DeviceServiceSchema]


class MachineAllocationSchema(BaseModel):
    host_name: str
    num_cpus: int
    total_memory: int
    cpu: ResourceUsageSchema
    memory: ResourceUsageSchema
    devices: list[DeviceAllocationViewSchema]


class ServiceDeviceAllocationSchema(BaseModel):
    uuid: str
    name: str
    index: int
    memory: int
    allocation_percentage: float


class ServiceAllocationSchema(BaseModel):
    service_id: int
    service_name: str
    runtime_status: RuntimeStatus
    cpu_count: int
    mem_size: int
    devices: list[ServiceDeviceAllocationSchema]
