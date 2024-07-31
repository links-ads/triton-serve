from pathlib import Path

import yaml


class TraefikConfigManager:
    def __init__(self, configs_path: Path):
        self.configs_path = configs_path

    def delete(self, service_name: str):
        """
        Deletes the traefik service file for the specified service.

        Args:
            service_name (str): The name of the service.
            configs_path (Path): The path to the traefik configs.

        Returns:
            `None`
        """
        yaml_file_name = self.configs_path / f"{service_name}.yaml"
        if yaml_file_name.exists():
            yaml_file_name.unlink()

    def add(self, service_prefix: str, service_name: str, api_keys: list[str]):
        """
        Updates the traefik config with the specified service.

        Args:
            service_prefix (str): The url prefix to use for the service.
            service_name (str): The name of the service.
            api_keys (list[str]): The list of api keys to use for the service.

        Returns:
            `None`

        """
        prefix_name = f"{service_prefix}/{service_name}"
        path_prefix = f"PathPrefix(`{prefix_name}`)"
        service_url = f"http://{service_name}:8000"

        yaml_file_name = self.configs_path / f"{service_name}.yaml"
        raw_data = {
            "http": {
                "services": {service_name: {"loadBalancer": {"servers": [{"url": service_url}]}}},
                "middlewares": {
                    f"{service_name}_stripprefix": {"stripPrefix": {"prefixes": [prefix_name]}},
                    f"{service_name}_auth": {
                        "plugin": {
                            "traefik-api-key-middleware": {
                                "authenticationHeader": True,
                                "authenticationheaderName": "X-API-Key",
                                "keys": api_keys,
                            }
                        }
                    },
                },
                "routers": {
                    service_name: {
                        "rule": path_prefix,
                        "entryPoints": ["http"],
                        "middlewares": [
                            f"{service_name}_auth@file",
                            f"{service_name}_stripprefix@file",
                        ],
                        "service": service_name,
                    }
                },
            }
        }

        with open(yaml_file_name, "w") as file:
            yaml.dump(raw_data, file)
