import docker
from contextlib import asynccontextmanager
from fastapi import FastAPI
from triton_serve.database.session import engine, SessionLocal
from triton_serve.database import models
from triton_serve.config import get_settings
from triton_serve.utils.utils import get_machine_info, list_gpus
from triton_serve.database.schemas import Machine, Device


def get_db():
    """Yields a database session safely.

    :yield: database session
    :rtype: Iterator[Session]
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def check_resources():
    """Check if the resources saved in the database match the current resources."""
    db = next(get_db())
    # check if machine exists
    machine = db.query(models.Machine).first()
    if machine is None:
        print("No machine found in the database")
        return False
    # check if machine resources match
    num_cpus, total_mem = get_machine_info()
    if machine.num_cpus != num_cpus or machine.total_memory != total_mem:
        print("The machine resources saved in the database do not match the current resources")
        return False
    # retrieve devices and check if they match
    devices = db.query(models.Device).filter(models.Device.machine.has(host_id=machine.host_id)).all()
    gpus = list_gpus()
    if len(devices) != len(gpus):
        print("The number of devices saved in the database do not match the current resources")
        return False
    for device in devices:
        # for each device in the gpus dict, check if there is a match with the device in the db, if there is not gpu = None => return False
        gpu = next((gpu for gpu in gpus if gpu["uuid"] == device.uuid), None)
        if gpu is None:
            print("The devices saved in the database do not match the current resources")
            return False
        # check if the device in the db matches the device in the gpus dict
        if device.name != gpu["name"] or device.memory != gpu["memory"] or device.index != gpu["index"]:
            print(gpu)
            print(device.name, device.memory, device.index)
            print("The devices saved in the database do not match the current resources")
            return False
    return True


@asynccontextmanager
async def lifespan(app: FastAPI):
    # create db tables if they do not exist
    models.Base.metadata.create_all(bind=engine)
    # assert that the resources saved in the db match the current resources, if not, crash
    assert check_resources(), "The resources saved in the database do not match the current resources"
    yield


def docker_client() -> docker.DockerClient:
    """Yields a docker client API instance safely.

    :return: docker client instance
    :rtype: docker.DockerClient
    :yield: docker client, useful to interact with the system
    :rtype: Iterator[docker.DockerClient]
    """
    client = docker.from_env()
    try:
        yield client
    finally:
        client.close()
