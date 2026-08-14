from types import SimpleNamespace

from pydantic import SecretStr

from triton_serve.builder.registry import StaticTokenAuth, auth_config, image_ref, pull_auth, push_auth

HASH = "0123456789abcdef" * 4


def _settings():
    return SimpleNamespace(
        registry_url="ghcr.io",
        registry_namespace="links-ads",
        registry_image_name="serve-runtime",
        registry_push_username="serve-bot",
        registry_push_token=SecretStr("push-secret"),
        registry_pull_username="serve-bot",
        registry_pull_token=SecretStr("pull-secret"),
    )


def test_image_ref_uses_the_short_hash():
    assert image_ref(_settings(), HASH) == "ghcr.io/links-ads/serve-runtime:0123456789ab"


def test_push_and_pull_use_distinct_tokens():
    assert push_auth(_settings()).credentials() == ("serve-bot", "push-secret")
    assert pull_auth(_settings()).credentials() == ("serve-bot", "pull-secret")


def test_auth_config_is_the_docker_shape():
    assert auth_config(StaticTokenAuth("u", SecretStr("t"))) == {"username": "u", "password": "t"}


def test_repr_does_not_leak_the_token():
    assert "t0ps3cret" not in repr(StaticTokenAuth("u", SecretStr("t0ps3cret")))
