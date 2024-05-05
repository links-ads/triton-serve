#!/bin/bash

# Build the Triton image with a specific name
docker build . \
    --build-arg TRITON_VERSION=${TRITON_VERSION:-23.07-py3} \
    -t ghcr.io/links-ads/serve-triton:${TRITON_VERSION:-23.07-py3} && \
    docker push ghcr.io/links-ads/serve-triton:${TRITON_VERSION:-23.07-py3}