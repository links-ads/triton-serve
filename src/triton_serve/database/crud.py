from sqlalchemy.orm import Session

from triton_serve.database import models, schemas


# MACHINES queries
def get_machine(db: Session, host_id: str):
    return db.query(models.Machine).filter(models.Machine.host_id == host_id).first()


def create_machine(db: Session, machine: schemas.MachineCreate):
    db_machine = models.Machine(**machine)
    db.add(db_machine)
    return db_machine


# DEVICES queries
def get_devices(db: Session, host_id: str):
    return db.query(models.Device).filter(models.Device.host_id == host_id).all()


def create_device(db: Session, device: schemas.DeviceCreate):
    db_device = models.Device(**device)
    db.add(db_device)
    return db_device


# SERVICES queries
def get_services(db: Session, host_id: str):
    return db.query(models.Service).filter(models.Service.host_id == host_id).all()


def create_service(db: Session, service: schemas.ServiceCreate):
    db_service = models.Service(**service)
    db.add(db_service)
    return db_service
