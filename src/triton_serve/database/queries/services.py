from triton_serve.database.connection import get_connection
from psycopg2 import extras
from typing import List
import psycopg2


def create_service(
    service_name: str,
    models: List[str],
    created_at: str,
    cursor: psycopg2.extensions.cursor,
    gpu_requested: bool = False,
):
    if gpu_requested:
        sql_check_free_gpus = "SELECT d.device_id, d.index \
        FROM public.devices d \
        WHERE d.device_id NOT IN \
        (SELECT s.assigned_device \
        FROM public.services s \
        WHERE s.assigned_device IS NOT NULL);"
        cursor.execute(sql_check_free_gpus)
        free_gpus = cursor.fetchall()

        if len(free_gpus) == 0:
            raise ValueError("No free GPUs available")

        # take the first free gpu
        gpu_id, gpu_index = free_gpus[0]["device_id"], free_gpus[0]["index"]
    else:
        gpu_id = None
        gpu_index = None
    sql_insert = "INSERT INTO public.services\
        (service_name, assigned_device, models, created_at)\
        VALUES(%s, %s, %s, %s);"
    cursor.execute(
        sql_insert,
        (
            service_name,
            gpu_id,
            models,
            created_at,
        ),
    )

    return gpu_index
