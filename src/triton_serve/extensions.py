import docker


def docker_client() -> docker.DockerClient:
    """Yields a docker client API instance safely.

    :return: docker client instance
    :rtype: docker.DockerClient
    :yield: docker client, useful to interact with the system
    :rtype: Iterator[docker.DockerClient]
    """
    client = docker.from_env()
    try:
        yield client
    finally:
        client.close()
