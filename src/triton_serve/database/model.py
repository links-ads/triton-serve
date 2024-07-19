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
)
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.orm import relationship

from triton_serve.database.manage import Base

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

utcnow = partial(datetime.now, tz=timezone.utc)


class Machine(Base):
    __tablename__ = "machines"
    host_id = Column(Integer, primary_key=True, autoincrement=True)
    host_name = Column(String, nullable=False)
    num_cpus = Column(Integer, nullable=False, default=0)
    total_memory = Column(Integer, nullable=False, default=0)
    devices = relationship("Device", back_populates="machine")


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


class Model(Base):
    __tablename__ = "models"
    model_name = Column(String, nullable=False)
    model_version = Column(Integer, nullable=False)
    model_uri = Column(String, nullable=False)
    model_type = Column(Enum(ModelType), nullable=False, default=ModelType.UNK)
    created_at = Column(DateTime(timezone=True), default=utcnow, nullable=False)
    updated_at = Column(DateTime(timezone=True), default=utcnow, nullable=False)
    source = Column(String, nullable=True)
    dependencies = Column(ARRAY(String), nullable=True, default=None)

    __table_args__ = (
        PrimaryKeyConstraint("model_name", "model_version", name="model_name_version"),
        CheckConstraint("model_version > 0", name="model_version_positive"),
    )


class Device(Base):
    __tablename__ = "devices"
    uuid = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    memory = Column(Integer, nullable=False)
    index = Column(Integer, nullable=False)
    host_id = Column(Integer, ForeignKey("machines.host_id"), nullable=False)
    machine = relationship("Machine", back_populates="devices")
    allocations = relationship("DeviceAllocation", back_populates="device")


class DeviceAllocation(Base):
    __tablename__ = "device_allocations"
    id = Column(Integer, primary_key=True)
    device_id = Column(String, ForeignKey("devices.uuid"))
    service_id = Column(Integer, ForeignKey("services.service_id"))
    allocation_percentage = Column(Float, nullable=False)
    allocated_at = Column(DateTime(timezone=True), default=utcnow, nullable=False)
    deallocated_at = Column(DateTime(timezone=True), default=None, nullable=True)

    device = relationship("Device", back_populates="allocations")
    service = relationship("Service", back_populates="devices")


class ServiceStatus(enum.Enum):
    STARTING = "starting"
    ACTIVE = "active"
    ERROR = "error"
    STOPPED = "stopped"
    DELETED = "deleted"


class Service(Base):
    __tablename__ = "services"
    service_id = Column(Integer, primary_key=True, autoincrement=True)
    service_name = Column(String, nullable=False)
    service_image = Column(String, nullable=False)
    container_id = Column(String, nullable=True)
    container_status = Column(Enum(ServiceStatus), nullable=False, default=ServiceStatus.STARTING)
    created_at = Column(DateTime(timezone=True), default=utcnow, nullable=False)
    deleted_at = Column(DateTime(timezone=True), nullable=True, default=None)
    cpu_count = Column(Integer, nullable=False)
    shm_size = Column(Integer, nullable=False)
    mem_size = Column(Integer, nullable=False)

    models = relationship("Model", secondary=model_mapping, backref="services")
    devices = relationship("DeviceAllocation", back_populates="service")

    __table_args__ = (Index("service_name_idx", "service_name", unique=True, postgresql_where=(deleted_at.is_(None))),)
