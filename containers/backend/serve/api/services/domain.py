import logging
import shutil
from typing import List, Optional
import yaml

from fastapi import HTTPException
from serve.api.models.dto import Model
from serve.common import utils

import docker
from serve.common.constants import BASE_PATH, LOCAL_MODEL_REPOSITORY


def create_container(client: docker.DockerClient, image, name, volumes, command, labels):
    """Creates a triton docker container.

    :param client: docker client
    :type client: docker.DockerClient
    :return: a container entity
    :rtype: Container
    """

    container = client.containers.create(
        image=image,
        name=name,
        volumes=volumes,
        command=command,
        labels=labels
    )

    return container


def spawn_triton_container(name: str, models: List[str]):
    client = docker.from_env()
    image = "nvcr.io/nvidia/tritonserver:23.07-py3"
    name = name
    volumes = {
        LOCAL_MODEL_REPOSITORY: {
            "bind": "/models",
            "mode": "ro"
        }
    }
    # create a string that containes the name of the mdels to be loaded prefixed by --load-model=
    models_string = ""
    for model in models:
        models_string += "--load-model=" + model + " "
    command = "tritonserver --model-repository=/models --model-control-mode=explicit " + models_string
    labels = {
        "sablier.enable": "true",
        "sablier.group": "tritons"
    }
    container = create_container(client, image, name, volumes, command, labels)
    container.start()
    return container


def update_traefik_config(name: str):
    service_name = name
    url_name = "http://" + name + ":8000"
    display_name = "Service: " + name
    prefix_name = "/" + name
    pathPrefix_name = 'PathPrefix(`/' + name + '`))'
    
    yaml_file_name = str(BASE_PATH) + r'/config_files/' + name + '_config.yaml'

    yaml_to_write = {'http':
                     {'services':
                      {service_name: {'loadBalancer':
                                      {'servers':
                                       [{'url': url_name}]}}},
                      
                      'middlewares':
                      {'my-sablier':
                       {'plugin':
                        {'sablier':
                         {'dynamic':
                          {'displayName': display_name,
                           'refreshFrequency': '5s',
                           'showDetails': 'true',
                           'theme': 'ghost'},
                          'group': 'tritons',
                          'names': service_name,
                          'sablierUrl': 'http://sablier:10000',
                          'sessionDuration': '60s'}}},
                       'stripPrefix':
                              {
                              'stripPrefix':
                                {'prefixes': [prefix_name]}}
                       },

                      
                      'routers':
                      {service_name:
                       {'rule': pathPrefix_name,
                        'entryPoints': ['http'],
                        'middlewares': ['my-sablier@file', 'stripPrefix@file'],
                        'service': service_name}}}}
    
    with open(yaml_file_name, 'w') as file:
        documents = yaml.dump(yaml_to_write, file)
    
    
        


def post_service(name: str, models: List[str]):
    """Creates a triton docker container loading the models specified in the models list.

    :param name: name of the service
    :type name: str
    :param models: list of models to be loaded
    :type models: List[str]
    """

    spawn_triton_container(name, models)
    update_traefik_config(name)
    return None
