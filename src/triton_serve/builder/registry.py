from dataclasses import dataclass
from typing import Protocol

from pydantic import SecretStr

from triton_serve.config.schema import AppSettings


class RegistryAuth(Protocol):
    """Indirection over registry credentials, so a static token can later become a refreshing one."""

    def credentials(self) -> tuple[str, str]:
        """Returns (username, token), refreshing if the implementation needs to."""
        ...


@dataclass(frozen=True)
class StaticTokenAuth:
    username: str
    token: SecretStr

    def credentials(self) -> tuple[str, str]:
        return self.username, self.token.get_secret_value()


def push_auth(settings: AppSettings) -> RegistryAuth:
    return StaticTokenAuth(settings.registry_push_username, settings.registry_push_token)


def pull_auth(settings: AppSettings) -> RegistryAuth:
    return StaticTokenAuth(settings.registry_pull_username, settings.registry_pull_token)


def auth_config(auth: RegistryAuth) -> dict[str, str]:
    """Adapts a provider to the dict shape docker-py's push/pull expect."""
    username, token = auth.credentials()
    return {"username": username, "password": token}


def image_ref(settings: AppSettings, image_hash: str) -> str:
    """The pullable reference for a built image, tagged by the truncated content hash."""
    return f"{settings.registry_url}/{settings.registry_namespace}/{settings.registry_image_name}:{image_hash[:12]}"
