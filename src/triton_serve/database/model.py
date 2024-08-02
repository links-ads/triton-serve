import enum
from datetime import datetime, timezone
from functools import partial

from sqlalchemy import (
    CheckConstraint,
    Column,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    ForeignKeyConstraint,
    Index,
    Integer,
    PrimaryKeyConstraint,
    String,
    Table,
    Text,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.orm import declarative_base, relationship

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


# Association table for many-to-many relationship between APIKey and Service
key_service_association = Table(
    "key_service_association",
    Base.metadata,
    Column("api_key_id", Integer, ForeignKey("api_keys.key_id")),
    Column("service_id", Integer, ForeignKey("services.service_id")),
    PrimaryKeyConstraint("api_key_id", "service_id", name="api_key_service"),
)


class KeyType(enum.Enum):
    ADMIN = "admin"
    USER = "user"
    SERVICE = "service"


class APIKey(Base):
    __tablename__ = "api_keys"

    key_id = Column(Integer, primary_key=True)
    key_type = Column(Enum(KeyType), nullable=False)
    value = Column(String, unique=True, nullable=False)
    project = Column(String, nullable=False)
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), default=utcnow)
    expires_at = Column(DateTime(timezone=True))

    services = relationship("Service", secondary=key_service_association)


model_mapping = Table(
    "model_mapping",
    Base.metadata,
    Column("service_id", Integer, ForeignKey("services.service_id", ondelete="CASCADE")),
    Column("model_name", String, nullable=False),
    Column("model_version", Integer, nullable=False),
    PrimaryKeyConstraint("service_id", "model_name", "model_version", name="service_model"),
    ForeignKeyConstraint(
        ["model_name", "model_version"],
        ["models.model_name", "models.model_version"],
        ondelete="CASCADE",
    ),
)


class Machine(Base):
    __tablename__ = "machines"
    host_id = Column(Integer, primary_key=True, autoincrement=True)
    host_name = Column(String, nullable=False)
    num_cpus = Column(Integer, nullable=False, default=0)
    total_memory = Column(Integer, nullable=False, default=0)
    devices = relationship("Device", back_populates="machine")


class Device(Base):
    __tablename__ = "devices"
    uuid = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    memory = Column(Integer, nullable=False)
    index = Column(Integer, nullable=False)
    host_id = Column(Integer, ForeignKey("machines.host_id"), nullable=False)
    machine = relationship("Machine", back_populates="devices")
    allocations = relationship("DeviceAllocation", back_populates="device")


class Model(Base):
    __tablename__ = "models"
    model_name = Column(String, nullable=False)
    model_version = Column(Integer, nullable=False)
    model_uri = Column(String, nullable=False)
    model_type = Column(Enum(ModelType), nullable=False, default=ModelType.UNK)
    created_at = Column(DateTime(timezone=True), default=utcnow, nullable=False)
    updated_at = Column(DateTime(timezone=True), default=utcnow, nullable=False)
    source = Column(String, nullable=True)
    dependencies = Column(ARRAY(String), nullable=True, default=[])

    __table_args__ = (
        PrimaryKeyConstraint("model_name", "model_version", name="model_name_version"),
        CheckConstraint("model_version > 0", name="model_version_positive"),
    )


class ServiceResources(Base):
    __tablename__ = "service_resources"
    service_id = Column(Integer, ForeignKey("services.service_id"), primary_key=True)
    cpu_count = Column(Integer, nullable=False)
    shm_size = Column(Integer, nullable=False)
    mem_size = Column(Integer, nullable=False)
    environment_variables = Column(JSONB, nullable=True)

    service = relationship("Service", back_populates="resources")

    __table_args__ = (
        CheckConstraint("cpu_count > 0", name="positive_cpu_count"),
        CheckConstraint("shm_size > 0", name="positive_shm_size"),
        CheckConstraint("mem_size > 0", name="positive_mem_size"),
    )


class Service(Base):
    __tablename__ = "services"
    service_id = Column(Integer, primary_key=True, autoincrement=True)
    service_name = Column(String, nullable=False)
    service_image = Column(String, nullable=False)
    container_id = Column(String, nullable=True)
    container_status = Column(Enum(ServiceStatus), nullable=False, default=ServiceStatus.STARTING)
    created_at = Column(DateTime(timezone=True), default=utcnow, nullable=False)
    deleted_at = Column(DateTime(timezone=True), nullable=True, default=None)
    inactivity_timeout = Column(Integer, nullable=False, default=3600)  # in seconds
    priority = Column(Integer, nullable=False, default=0)  # higher number means higher priority
    last_active_time = Column(DateTime(timezone=True), nullable=True)

    models = relationship("Model", secondary=model_mapping, backref="services")
    device_allocations = relationship("DeviceAllocation", back_populates="service")
    resources = relationship("ServiceResources", uselist=False, back_populates="service", cascade="all, delete-orphan")

    __table_args__ = (
        Index("service_name_idx", "service_name", unique=True, postgresql_where=(deleted_at.is_(None))),
        CheckConstraint("inactivity_timeout >= 0", name="non_negative_timeout"),
        CheckConstraint("priority >= 0", name="non_negative_priority"),
    )


class DeviceAllocation(Base):
    __tablename__ = "device_allocations"
    id = Column(Integer, primary_key=True)
    device_id = Column(String, ForeignKey("devices.uuid"))
    service_id = Column(Integer, ForeignKey("services.service_id"))
    allocation_percentage = Column(Float, nullable=False)

    device = relationship("Device", back_populates="allocations")
    service = relationship("Service", back_populates="device_allocations")

    __table_args__ = (
        CheckConstraint(
            "allocation_percentage > 0 AND allocation_percentage <= 100", name="valid_allocation_percentage"
        ),
    )
