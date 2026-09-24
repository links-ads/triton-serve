from pathlib import Path

import yaml


class TraefikConfigManager:
    def __init__(self, configs_path: Path):
        self.configs_path = configs_path

    def names(self) -> set[str]:
        """Reports the service names that currently have a config file on disk.

        Returns:
            set[str]: one name per managed config file.
        """
        return {path.stem for path in self.configs_path.glob("*.yaml")}

    def delete(self, service_name: str):
        """
        Deletes the traefik service file for the specified service.

        Args:
            service_name (str): The name of the service.

        Returns:
            `None`
        """
        yaml_file_name = self.configs_path / f"{service_name}.yaml"
        if yaml_file_name.exists():
            yaml_file_name.unlink()

    def add(self, service_prefix: str, service_name: str):
        """
        Updates the traefik config with the specified service.

        Args:
            service_prefix (str): The url prefix to use for the service.
            service_name (str): The name of the service.

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
                    f"{service_name}-stripprefix": {
                        "stripPrefix": {"prefixes": [prefix_name]},
                    },
                    f"{service_name}-forward": {
                        "forwardAuth": {
                            "address": f"http://backend:5000/status/{service_name}",
                        },
                    },
                },
                "routers": {
                    service_name: {
                        "rule": path_prefix,
                        "entryPoints": ["http"],
                        "middlewares": [
                            f"{service_name}-forward@file",  # the backend authorizes the key and reports readiness
                            f"{service_name}-stripprefix@file",
                        ],
                        "service": service_name,
                    }
                },
            }
        }

        with open(yaml_file_name, "w") as file:
            yaml.dump(raw_data, file)
