import contextlib

from sqlalchemy import Connection, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

Base = declarative_base()


class DatabaseManager:
    def __init__(self):
        self._engine = None
        self._sessionmaker = None

    def init(self, database_url: str):
        self._engine = create_engine(database_url)
        self._sessionmaker = sessionmaker(
            bind=self._engine,
            autocommit=False,
            autoflush=False,
            expire_on_commit=False,
        )

    def close(self):
        if self._engine is not None:
            self._engine.dispose()
        self._engine = None
        self._sessionmaker = None

    @contextlib.contextmanager
    def connect(self):
        if self._engine is None:
            raise Exception("DatabaseSessionManager is not initialized")
        with self._engine.begin() as connection:
            try:
                yield connection
            except Exception:
                connection.rollback()
                raise

    @contextlib.contextmanager
    def session(self):
        if self._sessionmaker is None:
            raise Exception("DatabaseSessionManager is not initialized")
        session = self._sessionmaker()
        try:
            yield session
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    # Used for testing
    def create_all(self, connection: Connection):
        connection.run_sync(Base.metadata.create_all)

    def drop_all(self, connection):
        connection.run_sync(Base.metadata.drop_all)


database_manager = DatabaseManager()
