import logging
from contextlib import asynccontextmanager
from importlib.metadata import version

import uvicorn
from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

from triton_serve.api import allocations, auth, models, operations, services
from triton_serve.api.services.domain import rebuild_service_config
from triton_serve.config import AppSettings, get_traefik
from triton_serve.database import database_manager
from triton_serve.database.model import Service
from triton_serve.database.validation import check_resources

log = logging.getLogger(uvicorn.__name__)


def sync_traefik_configs(session, settings: AppSettings) -> None:
    """Rebuilds every non-deleted service's Traefik config from database truth."""
    traefik = get_traefik()
    services = session.query(Service).filter(Service.deleted_at.is_(None)).all()
    for service in services:
        rebuild_service_config(session, traefik, service, settings.service_prefix, settings.api_keys)


def create_app(settings: AppSettings, init_database: bool = True) -> FastAPI:
    """Factory method that creates a new FastAPI application.

    :return: configured FastAPI instance
    :rtype: FastAPI
    """
    if init_database:
        database_manager.init(settings.database_url)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """
        Context manager that handles the app lifespan events.
        Checks if the resources saved in the database match the current resources,
        and closes the database connection when the app is shutting down.
        """
        with database_manager.session() as session:
            try:
                check_resources(session=session)
            except AssertionError as e:
                log.warning("Validation error at startup: %s", str(e))
                log.warning("Triton Serve may need to be reinitialized")
            try:
                sync_traefik_configs(session, settings)
            except Exception as e:
                log.warning("Failed to sync Traefik configs at startup: %s", e)
        yield
        database_manager.close()

    app = FastAPI(
        title=settings.api_title,
        version=version("triton_serve"),
        description=settings.api_description,
        root_path=settings.api_root_path,
        lifespan=lifespan,
    )
    register_middlewares(app)
    register_routers(app)
    return app


def register_middlewares(app: FastAPI):
    """Registers middlewares to the main application instance.

    :param app: app instance
    :type app: FastAPI
    """
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


def register_routers(app: FastAPI):
    """Registers all the available submodules to the main application.

    :param app: FastAPI instance
    :type app: FastAPI
    """
    app.include_router(models.router)
    app.include_router(services.router)
    app.include_router(auth.router)
    app.include_router(operations.router)
    app.include_router(allocations.router)
