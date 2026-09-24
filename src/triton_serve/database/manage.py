import contextlib

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker


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
    def connect(self, isolation_level: str | None = None):
        if self._engine is None:
            raise Exception("DatabaseSessionManager is not initialized")
        engine = self._engine.execution_options(isolation_level=isolation_level) if isolation_level else self._engine
        with engine.begin() as connection:
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

    @contextlib.contextmanager
    def advisory_lock(self, key: int):
        """Holds a postgres advisory lock for one pass, yielding whether it was acquired.

        The connection is dedicated and AUTOCOMMIT on purpose: an ORM session releases its
        connection on every commit, which would strand the lock on a pooled connection and later
        unlock a different one, and a transactional connection would sit idle-in-transaction,
        pinning a snapshot for the whole pass.

        Args:
            key (int): the lock key; each caller owns a distinct one.

        Yields:
            bool: True while the lock is held, False when another pass already holds it.
        """
        with self.connect(isolation_level="AUTOCOMMIT") as connection:
            acquired = bool(connection.execute(text("SELECT pg_try_advisory_lock(:k)"), {"k": key}).scalar())
            try:
                yield acquired
            finally:
                if acquired:
                    connection.execute(text("SELECT pg_advisory_unlock(:k)"), {"k": key})
