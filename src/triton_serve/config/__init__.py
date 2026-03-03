from functools import lru_cache

from triton_serve.config.schema import AppSettings
from triton_serve.config.traefik import TraefikConfigManager
from triton_serve.storage import LocalModelStorage


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    """
    Istantiates the app settings, caching them for reuse.
    """
    return AppSettings()  # type: ignore


@lru_cache(maxsize=1)
def get_storage():
    """
    Istantiates the model storage, caching it for reuse.
    """
    settings = get_settings()
    if settings.storage_type == "local":
        return LocalModelStorage(settings.repository_path)
    else:
        raise NotImplementedError(f"Storage type {settings.storage_type} not implemented")


@lru_cache(maxsize=1)
def get_traefik():
    """
    Istantiates the traefik config manager, caching it for reuse.
    """
    settings = get_settings()
    return TraefikConfigManager(settings.configs_path)


__all__ = ["get_settings", "get_storage", "AppSettings"]
