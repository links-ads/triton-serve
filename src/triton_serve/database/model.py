import enum
from datetime import datetime, timezone
from functools import partial

from sqlalchemy import (
    CheckConstraint,
    Column,
    DateTime,
    ForeignKey,
    Index,
    PrimaryKeyConstraint,
    String,
    Table,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.orm import Mapped, declarative_base, mapped_column, relationship

Base = declarative_base()

utcnow = partial(datetime.now, tz=timezone.utc)


class ModelType(enum.Enum):
    UNK = "unknown"
    TENSORRT = "tensorrt"
    ONNX = "onnx"
    TORCHSCRIPT = "torchscript"
    TENSORFLOW = "tensorflow"
    OPENVINO = "openvino"
    PYTHON = "python"
    DALI = "dali"
    ENSEMBLE = "ensemble"


class ServiceStatus(enum.Enum):
    STARTING = "starting"
    ACTIVE = "active"
    ERROR = "error"
    STOPPED = "stopped"
    DELETED = "deleted"


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
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
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
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    deleted_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=True, default=None)
    source: Mapped[str] = mapped_column(nullable=True)
    dependencies: Mapped[list[str]] = mapped_column(ARRAY(String), nullable=True, default=[])
    version_policy: Mapped[dict] = mapped_column(JSONB, nullable=True)
    versions: Mapped[list["ModelVersion"]] = relationship("ModelVersion")

    __table_args__ = (Index("model_name_idx", "model_name", unique=True, postgresql_where=(deleted_at.is_(None))),)


class ModelVersion(Base):
    __tablename__ = "model_versions"

    model_id: Mapped[int] = mapped_column(ForeignKey(Model.model_id), primary_key=True)
    version_id: Mapped[int] = mapped_column(primary_key=True)
    model_uri: Mapped[str] = mapped_column(nullable=False)

    __table_args__ = (CheckConstraint("version_id > 0", name="version_positive"),)


class Service(Base):
    __tablename__ = "services"

    service_id: Mapped[int] = mapped_column(primary_key=True)
    service_name: Mapped[str] = mapped_column(nullable=False)
    service_image: Mapped[str] = mapped_column(nullable=False)
    container_id: Mapped[str] = mapped_column(nullable=True)
    container_status: Mapped[ServiceStatus] = mapped_column(nullable=False, default=ServiceStatus.STARTING)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    deleted_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=True, default=None)
    last_active_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    inactivity_timeout: Mapped[int] = mapped_column(nullable=False, default=3600)  # 1 hour
    priority: Mapped[int] = mapped_column(nullable=False)  # 0 is the lowest priority

    models: Mapped[list["Model"]] = relationship(secondary=model_service_association, backref="services")
    device_allocations: Mapped[list["DeviceAllocation"]] = relationship(back_populates="service")
    resources: Mapped["ServiceResources"] = relationship(back_populates="service")

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
