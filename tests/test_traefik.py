import pytest
import yaml

from triton_serve.config.traefik import TraefikConfigManager


@pytest.fixture
def traefik(tmp_path):
    return TraefikConfigManager(tmp_path)


def _middlewares(config_path) -> dict:
    with open(config_path) as file:
        config = yaml.safe_load(file)
    return config["http"]["middlewares"]


def test_add_writes_no_key_material(traefik):
    traefik.add(service_prefix="", service_name="svc")

    middlewares = _middlewares(traefik.configs_path / "svc.yaml")

    assert set(middlewares) == {"svc-stripprefix", "svc-forward"}
    assert "plugin" not in yaml.dump(middlewares)


def test_add_keeps_the_forward_hook_before_stripprefix(traefik):
    """forwardAuth is now the only thing enforcing the ACL, so it must run before proxying."""
    traefik.add(service_prefix="", service_name="svc")

    with open(traefik.configs_path / "svc.yaml") as file:
        router = yaml.safe_load(file)["http"]["routers"]["svc"]

    assert router["middlewares"] == ["svc-forward@file", "svc-stripprefix@file"]
