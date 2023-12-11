import psycopg2
from triton_serve.config import get_settings
from functools import lru_cache


@lru_cache(maxsize=1)
def get_connection():
    config = get_settings()
    return DB_Connection(config).connection


class DB_Connection:
    def __init__(self, config):
        try:
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
