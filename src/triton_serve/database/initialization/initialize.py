import psycopg2
from psycopg2 import sql
from pathlib import Path
from triton_serve.database.connection import get_connection
import os
import psutil
from triton_serve.config import get_settings
from triton_serve.database.initialization.checks import get_machine_info, list_gpus
import logging

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


def create_tables(script_path, cursor):
    log.info("Creating database and tables...")
    # Read the DDL script
    with open(script_path, "r") as file:
        ddl_script = file.read()

    # Connect to the PostgreSQL database
    try:
        # Execute the DDL script
        cursor.execute(ddl_script)

        log.info("Database and tables created successfully")
    except Exception as e:
        print(f"Error executing DDL script: {e}")


def populate_machines(machine_name, cursor):
    log.info("Populating machines table...")
    insert_sql = "INSERT INTO public.machines\
                (host_name, num_cpus, total_memory) "
    values_sql = "VALUES (%s, %s, %s) "
    return_sql = "RETURNING host_id;"
    insert_sql = insert_sql + values_sql + return_sql

    num_cpus, total_memory = get_machine_info()
    values = (machine_name, num_cpus, total_memory)
    cursor.execute(insert_sql, values)
    host_id = cursor.fetchone()[0]
    log.info("Machines table populated successfully")
    return host_id


def populate_devices(cursor, host_id):
    log.info("Populating devices table...")
    insert_sql = "INSERT INTO public.devices\
                (host_id, uuid, name, total_memory, index) "
    values_sql = "VALUES (%s, %s, %s, %s, %s);"
    insert_sql = insert_sql + values_sql

    gpus = list_gpus()

    for gpu in gpus:
        values = (host_id, gpu.device_id, gpu.name, gpu.memory, gpu.index)
        cursor.execute(insert_sql, values)

    log.info("Devices table populated successfully")


def initialize(connection, config):
    try:
        cursor = connection.cursor()

        # DATABASE AND TABLE CREATION
        current_dir = Path(os.path.dirname(os.path.realpath(__file__)))
        ddl_script_path = current_dir / "ddl.sql"
        create_tables(ddl_script_path, cursor)

        # POPULATE MACHINES WITH THE CURRENT MACHINE RESOURCES
        machine_name = "loki"
        host_id = populate_machines(machine_name, cursor)

        # POPULATE DEVICES WITH THE CURRENT GPU RESOURCES
        populate_devices(cursor, host_id)

        # COMMIT CHANGES
        log.info("Committing changes...")
        connection.commit()
        log.info("Changes committed successfully")
        cursor.close()
        connection.close()

    except:
        log.error("Error initializing database")
        connection.rollback()
        cursor.close()
        connection.close()
        raise


if __name__ == "__main__":
    connection = get_connection()
    config = get_settings()

    initialize(connection, config)
