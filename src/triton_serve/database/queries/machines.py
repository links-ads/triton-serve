from triton_serve.database.connection import get_connection
from psycopg2 import extras


def get_machine_resources():
    cursor = get_connection().cursor(cursor_factory=extras.RealDictCursor)
    cursor.execute("SELECT * FROM machines")
    resources = cursor.fetchone()
    return resources
