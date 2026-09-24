import pytest
import yaml

from triton_serve.config.traefik import TraefikConfigManager


@pytest.fixture
def traefik(tmp_path):
    return TraefikConfigManager(tmp_path)


def _config(config_path) -> dict:
    with open(config_path) as file:
        return yaml.safe_load(file)["http"]


def test_add_writes_no_key_material(traefik):
    traefik.add(service_prefix="", service_name="svc")

    middlewares = _config(traefik.configs_path / "svc.yaml")["middlewares"]

    assert set(middlewares) == {"svc-stripprefix", "svc-forward"}
    # the names alone would not catch a plugin body reappearing inside one of the two we allow
    assert "plugin" not in yaml.dump(middlewares)


def test_add_keeps_the_forward_hook_before_stripprefix(traefik):
    """forwardAuth is the only thing enforcing the ACL, so it must run before proxying."""
    traefik.add(service_prefix="", service_name="svc")

    router = _config(traefik.configs_path / "svc.yaml")["routers"]["svc"]

    assert router["middlewares"] == ["svc-forward@file", "svc-stripprefix@file"]
