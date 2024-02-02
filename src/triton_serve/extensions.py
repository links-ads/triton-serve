import docker

from triton_serve.database import database_manager


def get_db():
    """Yields a database session safely.

    :yield: database session
    :rtype: Iterator[Session]
    """
    with database_manager.session() as session:
        yield session


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
