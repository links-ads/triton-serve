from triton_serve.database.connection import get_connection
from triton_serve.database.dto import GPU
from psycopg2 import extras


def get_devices(host_id):
    connection = get_connection()
    cursor = connection.cursor(cursor_factory=extras.RealDictCursor)
    cursor.execute("SELECT * FROM devices WHERE host_id = (%s)", (host_id,))
    devices = cursor.fetchall()
    connection.commit()
    cursor.close()
    gpus = [GPU(**device) for device in devices]
    return gpus
