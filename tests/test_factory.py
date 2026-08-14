from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.exc import SQLAlchemyError

from triton_serve.factory import register_exception_handlers


def test_database_errors_return_an_opaque_500():
    """A driver message must never reach the client: it carries SQL, table names and row values."""
    app = FastAPI()
    register_exception_handlers(app)

    @app.get("/boom")
    def boom():
        raise SQLAlchemyError('relation "api_keys" does not exist: SELECT value FROM api_keys')

    response = TestClient(app, raise_server_exceptions=False).get("/boom")

    assert response.status_code == 500
    assert response.json() == {"detail": "Internal database error"}
