from functools import lru_cache

from triton_serve.config.schema import AppSettings
from triton_serve.storage import LocalModelStorage


@lru_cache(maxsize=1)
def get_settings():
    """
    Istantiates the app settings, caching them for reuse.
    """
    return AppSettings()


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


__all__ = ["get_settings", "get_storage", "AppSettings"]
