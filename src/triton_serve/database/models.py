from sqlalchemy import Boolean, Column, ForeignKey, Integer, String, ARRAY
from sqlalchemy.orm import relationship

from triton_serve.database.session import Base


class Machine(Base):
    __tablename__ = "machines"

    host_id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    host_name = Column(String)
    num_cpus = Column(Integer)
    total_memory = Column(Integer)

    devices = relationship("Device", back_populates="machine")


class Device(Base):
    __tablename__ = "devices"

    uuid = Column(String, primary_key=True, index=True)
    host_id = Column(Integer, ForeignKey("machines.host_id"))
    name = Column(String)
    memory = Column(Integer)
    index = Column(Integer)

    machine = relationship("Machine", back_populates="devices")
    service = relationship("Service", back_populates="device")


class Service(Base):
    __tablename__ = "services"

    service_name = Column(String, primary_key=True, index=True)
    models = Column(ARRAY(String))
    created_at = Column(Integer)
    assigned_device = Column(String, ForeignKey("devices.uuid"))

    device = relationship("Device", back_populates="service")
