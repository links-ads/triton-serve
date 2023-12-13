import psycopg2
from triton_serve.config import get_settings
from functools import lru_cache


def get_connection():
    config = get_settings()
    return DB_Connection(config).connection


def rollback(connection, cursor):
    if connection is not None:
        connection.rollback()
        connection.close()
    if cursor is not None:
        cursor.close()


class DB_Connection:
    def __new__(cls, config):
        if not hasattr(cls, "instance"):
            cls.instance = super(DB_Connection, cls).__new__(cls)
            cls.instance.connection = None
        return cls.instance

    def __init__(self, config):
        try:
            if self.connection is None or self.connection.closed != 0:
                self.connection = psycopg2.connect(
                    host=config.db_host,
                    database=config.db_name,
                    user=config.db_user,
                    password=config.db_password,
                    port=config.db_port,
                )
        except (Exception, psycopg2.DatabaseError) as error:
            print(error)
            self.connection = None
