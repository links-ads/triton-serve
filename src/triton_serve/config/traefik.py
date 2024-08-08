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
                    f"{service_name}-stripprefix": {
                        "stripPrefix": {"prefixes": [prefix_name]},
                    },
                    f"{service_name}-auth": {
                        "plugin": {
                            "traefik-api-key-middleware": {
                                "authenticationHeader": True,
                                "authenticationheaderName": "X-API-Key",
                                "removeHeadersOnSuccess": False,
                                "keys": api_keys,
                            }
                        }
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
                            f"{service_name}-auth@file",  # we first check that the key is associated with the service
                            f"{service_name}-forward@file",  # then we check that the service is running (if it is stopped, we start it)
                            f"{service_name}-stripprefix@file",  # we strip the prefix from the request
                        ],
                        "service": service_name,
                    }
                },
            }
        }

        with open(yaml_file_name, "w") as file:
            yaml.dump(raw_data, file)

    def add_service_key(self, service_name: str, key: str):
        """
        Adds a new API key to the service's configuration.

        Args:
            service_name (str): The name of the service.
            key (str): The API key to add.

        Returns:
            `None`
        """
        yaml_file_name = self.configs_path / f"{service_name}.yaml"
        if not yaml_file_name.exists():
            raise FileNotFoundError(f"Configuration file for service {service_name} not found.")

        with open(yaml_file_name) as file:
            config = yaml.safe_load(file)

        auth_middleware = config["http"]["middlewares"][f"{service_name}-auth"]["plugin"]["traefik-api-key-middleware"]
        if key not in auth_middleware["keys"]:
            auth_middleware["keys"].append(key)

        with open(yaml_file_name, "w") as file:
            yaml.dump(config, file)

    def remove_service_key(self, service_name: str, key: str):
        """
        Removes an API key from the service's configuration.

        Args:
            service_name (str): The name of the service.
            key (str): The API key to remove.

        Returns:
            `None`
        """
        yaml_file_name = self.configs_path / f"{service_name}.yaml"
        if not yaml_file_name.exists():
            raise FileNotFoundError(f"Configuration file for service {service_name} not found.")

        with open(yaml_file_name) as file:
            config = yaml.safe_load(file)

        auth_middleware = config["http"]["middlewares"][f"{service_name}-auth"]["plugin"]["traefik-api-key-middleware"]
        if key in auth_middleware["keys"]:
            auth_middleware["keys"].remove(key)

        with open(yaml_file_name, "w") as file:
            yaml.dump(config, file)
