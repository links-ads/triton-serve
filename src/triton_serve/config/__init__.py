from functools import lru_cache

from triton_serve.config.schema import AppSettings, StorageType
from triton_serve.config.traefik import TraefikConfigManager
from triton_serve.storage import LocalModelStorage, ModelStorage


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    """
    Istantiates the app settings, caching them for reuse.
    """
    return AppSettings()  # type: ignore


@lru_cache(maxsize=1)
def get_storage() -> ModelStorage:
    """
    Istantiates the model storage, caching it for reuse.
    """
    settings = get_settings()
    if settings.storage_type == StorageType.local:
        return LocalModelStorage(settings.repository_path, volume=settings.service_volume)
    # imported here so a local deployment never needs the azure extra installed
    from triton_serve.storage.azure import AzureModelStorage

    return AzureModelStorage(
        account=settings.azure_storage_account,
        container=settings.azure_storage_container,
        credential=settings.azure_storage_key.get_secret_value(),
        prefix=settings.azure_storage_prefix,
        stash_prefix=settings.azure_stash_prefix,
        endpoint=settings.azure_storage_endpoint,
    )


@lru_cache(maxsize=1)
def get_traefik():
    """
    Istantiates the traefik config manager, caching it for reuse.
    """
    settings = get_settings()
    return TraefikConfigManager(settings.configs_path)


__all__ = ["AppSettings", "get_settings", "get_storage"]
