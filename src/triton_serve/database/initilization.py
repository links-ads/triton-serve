import psycopg2
from triton_serve.config.schema import AppSettings

from triton_serve.database.schemas import DeviceCreate, MachineCreate
from triton_serve.database.models import Device, Machine
from triton_serve.utils.utils import get_machine_info, list_gpus

from triton_serve.database import models
from triton_serve.database.session import SessionLocal, engine

from triton_serve.config import get_settings
from triton_serve.extensions import get_db


def create_db(config: AppSettings, drop_db=False):
    """
    Create the database and tables

    :param config: the config object
    :param drop_db: whether to drop the database before creating it

    :return: None
    """
    # establishing the connection
    conn = psycopg2.connect(
        user=config.POSTGRES_USER,
        password=config.POSTGRES_PASSWORD,
        host=config.database_host,
        port=config.database_port,
    )
    conn.autocommit = True

    # Creating a cursor object using the cursor() method
    cursor = conn.cursor()

    # Preparing query to create a database
    if drop_db:
        cursor.execute(f"DROP DATABASE IF EXISTS {config.database_name};")
        print(f"Database {config.database_name} dropped successfully")
        conn.commit()

    cursor.execute(f"SELECT 1 FROM pg_catalog.pg_database WHERE datname = '{config.database_name}';")
    exists = cursor.fetchone()
    if not exists:
        cursor.execute(f"CREATE DATABASE {config.database_name}")
        print(f"Database {config.database_name} created successfully")
    else:
        print(f"Database {config.database_name} already exists, skipping creation")

    # create the tables
    models.Base.metadata.create_all(bind=engine)
    # Closing the connection
    conn.commit()
    cursor.close()
    conn.close()


def populate_db(db: SessionLocal):
    """
    Populate the database with the machine and gpu devices information

    :param db: the database session
    :return: None
    """
    num_cpus, total_mem = get_machine_info()
    machine = MachineCreate(host_name="loki", num_cpus=num_cpus, total_memory=total_mem)
    machine_db = Machine(**machine.model_dump())
    # add machine to db and get id
    db.add(machine_db)
    db.commit()
    db.refresh(machine_db)

    host_id = machine_db.host_id
    # add devices to db
    gpus = list_gpus()
    for gpu in gpus:
        gpu_to_create = DeviceCreate(
            uuid=gpu["uuid"],
            name=gpu["name"],
            memory=gpu["memory"],
            index=gpu["index"],
        )
        gpu_db = Device(**gpu_to_create.model_dump())
        gpu_db.host_id = host_id
        db.add(gpu_db)
        db.commit()
        db.refresh(gpu_db)


def initialize_db(config: AppSettings, db: SessionLocal):
    """
    Initialize the database

    :param config: the config object
    :param db: the database session
    :return: None
    """
    create_db(config, drop_db=False)
    populate_db(db)


if __name__ == "__main__":
    config = get_settings()
    db = next(get_db())
    initialize_db(config, get_db())
