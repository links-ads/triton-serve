import pytest
import yaml

from triton_serve.config.traefik import TraefikConfigManager


@pytest.fixture
def traefik(tmp_path):
    return TraefikConfigManager(tmp_path)


def _read_keys(config_path) -> list[str]:
    with open(config_path) as file:
        config = yaml.safe_load(file)
    name = config_path.stem
    middlewares = config["http"]["middlewares"]
    return middlewares[f"{name}-auth"]["plugin"]["traefik-api-key-middleware"]["keys"]


def test_add_writes_exactly_the_given_keys(traefik):
    traefik.add(service_prefix="", service_name="svc", api_keys=["k1", "k2"])
    assert _read_keys(traefik.configs_path / "svc.yaml") == ["k1", "k2"]
