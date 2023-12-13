from triton_serve.database.connection import get_connection
from psycopg2 import extras


def get_machine_resources():
    connection = get_connection()
    cursor = connection.cursor(cursor_factory=extras.RealDictCursor)
    cursor.execute("SELECT * FROM machines")
    resources = cursor.fetchone()
    connection.commit()
    cursor.close()
    return resources
