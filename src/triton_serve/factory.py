from fastapi import FastAPI, Security
from starlette.middleware.cors import CORSMiddleware

from triton_serve import __version__
from triton_serve.api import models, services
from triton_serve.config import AppSettings
from triton_serve.security import api_key_auth


def create_app(settings: AppSettings) -> FastAPI:
    """Factory method that creates a new FastAPI application.

    :return: configured FastAPI instance
    :rtype: FastAPI
    """
    app = FastAPI(title=settings.api_title, version=__version__, description=settings.api_description)
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
    app.include_router(models.router, tags=["models"], dependencies=[Security(api_key_auth)])
    app.include_router(services.router, tags=["services"], dependencies=[Security(api_key_auth)])
