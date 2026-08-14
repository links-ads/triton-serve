import enum
from datetime import datetime, timezone

from sqlalchemy import (
    CheckConstraint,
    Column,
    DateTime,
    Enum,
    ForeignKey,
    Index,
    PrimaryKeyConstraint,
    String,
    Table,
    Text,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.orm import Mapped, declarative_base, mapped_column, relationship

Base = declarative_base()


def timezone_aware_now() -> datetime:
    """The single source of "now" for the whole project.

    Deliberately not named `utcnow`: `datetime.utcnow` returns a *naive* timestamp, and every
    column here is `DateTime(timezone=True)`, so mixing the two silently produces comparisons
    between aware and naive datetimes.
    """
    return datetime.now(tz=timezone.utc)


class ModelType(enum.Enum):
    UNK = "unknown"
    TENSORRT = "tensorrt"
    ONNX = "onnxruntime_onnx"
    TORCHSCRIPT = "pytorch_libtorch"
    TENSORFLOW = "tensorflow"
    OPENVINO = "openvino"
    PYTHON = "python"
    DALI = "dali"
    ENSEMBLE = "ensemble"


class DesiredState(enum.Enum):
    AVAILABLE = "available"  # serve; reconciler may scale to 0 on idle and wakes on request
    SUSPENDED = "suspended"  # operator forced off; no auto-wake
    RETIRED = "retired"  # deleted; allocation released, traefik config removed


class RuntimeStatus(enum.Enum):
    READY = "ready"
    WARMING = "warming"
    IDLE = "idle"
    RECOVERING = "recovering"
    FAILED = "failed"
    SUSPENDED = "suspended"
    RETIRED = "retired"


class ImageStatus(enum.Enum):
    PENDING = "pending"
    BUILDING = "building"
    READY = "ready"
    FAILED = "failed"


class KeyType(enum.Enum):
    ADMIN = "admin"
    USER = "user"
    SERVICE = "service"


# Association table for many-to-many relationship between APIKey and Service
key_service_association = Table(
    "key_service_association",
    Base.metadata,
    Column("api_key_id", ForeignKey("api_keys.key_id", ondelete="CASCADE")),
    Column("service_id", ForeignKey("services.service_id", ondelete="CASCADE")),
    PrimaryKeyConstraint("api_key_id", "service_id", name="api_key_service"),
)

# Association table for many-to-many relationship between Model and Service
model_service_association = Table(
    "model_mapping",
    Base.metadata,
    Column("service_id", ForeignKey("services.service_id", ondelete="CASCADE")),
    Column("model_id", ForeignKey("models.model_id", ondelete="CASCADE")),
    PrimaryKeyConstraint("service_id", "model_id", name="service_model"),
)


class APIKey(Base):
    __tablename__ = "api_keys"

    key_id: Mapped[int] = mapped_column(primary_key=True)
    key_type: Mapped[KeyType] = mapped_column(nullable=False)
    value: Mapped[str] = mapped_column(unique=True, nullable=False)
    project: Mapped[str] = mapped_column(nullable=False)
    notes: Mapped[str] = mapped_column(nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=timezone_aware_now)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))

    services: Mapped[list["Service"]] = relationship(secondary=key_service_association)


class Machine(Base):
    __tablename__ = "machines"

    host_id: Mapped[int] = mapped_column(primary_key=True)
    host_name: Mapped[str] = mapped_column(nullable=False)
    num_cpus: Mapped[int] = mapped_column(nullable=False, default=0)
    total_memory: Mapped[int] = mapped_column(nullable=False, default=0)
    devices: Mapped[list["Device"]] = relationship(back_populates="machine")


class Device(Base):
    __tablename__ = "devices"

    uuid: Mapped[str] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(nullable=False)
    memory: Mapped[int] = mapped_column(nullable=False)
    index: Mapped[int] = mapped_column(nullable=False)
    host_id: Mapped[int] = mapped_column(ForeignKey("machines.host_id"), nullable=False)
    machine: Mapped["Machine"] = relationship(back_populates="devices")
    allocations: Mapped[list["DeviceAllocation"]] = relationship(back_populates="device")


class Model(Base):
    __tablename__ = "models"

    model_id: Mapped[int] = mapped_column(primary_key=True)
    model_name: Mapped[str] = mapped_column(nullable=False)
    model_type: Mapped[ModelType] = mapped_column(nullable=False, default=ModelType.UNK)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=timezone_aware_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=timezone_aware_now)
    deleted_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=True, default=None)
    source: Mapped[str] = mapped_column(nullable=True)
    dependencies: Mapped[list[str]] = mapped_column(ARRAY(String), nullable=True, default=[])
    system_dependencies: Mapped[list[str]] = mapped_column(
        ARRAY(String), nullable=False, default=list, server_default="{}"
    )
    version_policy: Mapped[dict] = mapped_column(JSONB, nullable=True)
    versions: Mapped[list["ModelVersion"]] = relationship("ModelVersion")

    __table_args__ = (Index("model_name_idx", "model_name", unique=True, postgresql_where=(deleted_at.is_(None))),)


class ModelVersion(Base):
    __tablename__ = "model_versions"

    model_id: Mapped[int] = mapped_column(ForeignKey(Model.model_id), primary_key=True)
    version_id: Mapped[int] = mapped_column(primary_key=True)
    model_uri: Mapped[str] = mapped_column(nullable=False)

    __table_args__ = (CheckConstraint("version_id > 0", name="version_positive"),)


class ServiceImage(Base):
    """A content-addressed runtime image. Rows are immutable: a changed spec is a different row."""

    __tablename__ = "service_images"

    image_hash: Mapped[str] = mapped_column(primary_key=True)
    image_ref: Mapped[str] = mapped_column(nullable=False)
    status: Mapped[ImageStatus] = mapped_column(
        Enum(ImageStatus, name="imagestatus", values_callable=lambda enum_cls: [e.value for e in enum_cls]),
        nullable=False,
        default=ImageStatus.PENDING,
        server_default=ImageStatus.PENDING.value,
    )
    managed: Mapped[bool] = mapped_column(nullable=False, default=True, server_default="true")
    base_image: Mapped[str] = mapped_column(nullable=False)
    apt_packages: Mapped[list[str]] = mapped_column(ARRAY(String), nullable=False, default=list)
    pip_packages: Mapped[list[str]] = mapped_column(ARRAY(String), nullable=False, default=list)
    build_log: Mapped[str | None] = mapped_column(Text, nullable=True, default=None)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=timezone_aware_now)
    built_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True, default=None)


class Service(Base):
    __tablename__ = "services"

    service_id: Mapped[int] = mapped_column(primary_key=True)
    service_name: Mapped[str] = mapped_column(nullable=False)
    service_image: Mapped[str] = mapped_column(nullable=False)
    image_hash: Mapped[str | None] = mapped_column(
        ForeignKey("service_images.image_hash"), nullable=True, default=None
    )
    container_id: Mapped[str | None] = mapped_column(default=None)
    # values_callable: the desiredstate/runtimestatus postgres enums store the lowercase .value
    # labels (see the lifecycle_redesign_expand migration), not the Python member names.
    desired_state: Mapped[DesiredState] = mapped_column(
        Enum(DesiredState, name="desiredstate", values_callable=lambda enum_cls: [e.value for e in enum_cls]),
        nullable=False,
        default=DesiredState.AVAILABLE,
        server_default=DesiredState.AVAILABLE.value,
    )
    runtime_status: Mapped[RuntimeStatus] = mapped_column(
        Enum(RuntimeStatus, name="runtimestatus", values_callable=lambda enum_cls: [e.value for e in enum_cls]),
        nullable=False,
        default=RuntimeStatus.WARMING,
        server_default=RuntimeStatus.WARMING.value,
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=timezone_aware_now)
    deleted_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=True, default=None)
    last_active_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    inactivity_timeout: Mapped[int] = mapped_column(nullable=False, default=3600)  # 1 hour
    priority: Mapped[int] = mapped_column(nullable=False)  # 0 is the lowest priority
    restart_attempts: Mapped[int] = mapped_column(nullable=False, default=0, server_default="0")
    last_attempt_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True, default=None)

    models: Mapped[list["Model"]] = relationship(secondary=model_service_association, backref="services")
    device_allocations: Mapped[list["DeviceAllocation"]] = relationship(back_populates="service")
    resources: Mapped["ServiceResources"] = relationship(back_populates="service")
    image: Mapped["ServiceImage | None"] = relationship()

    __table_args__ = (
        Index("service_name_idx", "service_name", unique=True, postgresql_where=(deleted_at.is_(None))),
        CheckConstraint("inactivity_timeout >= 0", name="non_negative_timeout"),
        CheckConstraint("priority >= 0", name="non_negative_priority"),
    )


class ServiceResources(Base):
    __tablename__ = "service_resources"

    service_id: Mapped[int] = mapped_column(ForeignKey("services.service_id"), primary_key=True)
    cpu_count: Mapped[int] = mapped_column(nullable=False)
    shm_size: Mapped[int] = mapped_column(nullable=False)
    mem_size: Mapped[int] = mapped_column(nullable=False)
    environment_variables: Mapped[dict] = mapped_column(JSONB, nullable=True)
    healthcheck: Mapped[dict | None] = mapped_column(JSONB, nullable=True, default=None)
    service: Mapped[Service] = relationship(back_populates="resources")

    __table_args__ = (
        CheckConstraint("cpu_count > 0", name="positive_cpu_count"),
        CheckConstraint("shm_size > 0", name="positive_shm_size"),
        CheckConstraint("mem_size > 0", name="positive_mem_size"),
    )


class DeviceAllocation(Base):
    __tablename__ = "device_allocations"

    allocation_id: Mapped[int] = mapped_column(primary_key=True)
    device_id: Mapped[str] = mapped_column(ForeignKey("devices.uuid"), nullable=False)
    service_id: Mapped[int] = mapped_column(ForeignKey("services.service_id"), nullable=False)
    allocation_percentage: Mapped[float] = mapped_column(nullable=False)

    device: Mapped[Device] = relationship(back_populates="allocations")
    service: Mapped[Service] = relationship(back_populates="device_allocations")

    __table_args__ = (
        CheckConstraint(
            "allocation_percentage > 0 AND allocation_percentage <= 100", name="valid_allocation_percentage"
        ),
    )


class KombuQueue(Base):
    __tablename__ = "kombu_queue"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(nullable=False)

    messages: Mapped[list["KombuMessage"]] = relationship("KombuMessage", back_populates="queue")


class KombuMessage(Base):
    __tablename__ = "kombu_message"

    id: Mapped[int] = mapped_column(primary_key=True)
    visible: Mapped[bool] = mapped_column()
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
    )
    payload: Mapped[str] = mapped_column()
    version: Mapped[int] = mapped_column()
    queue_id: Mapped[int] = mapped_column(ForeignKey("kombu_queue.id"))

    queue: Mapped["KombuQueue"] = relationship("KombuQueue", back_populates="messages")
