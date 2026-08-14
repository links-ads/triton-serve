# Triton Serve

A simple deployment framework based on [NVIDIA Triton Inference Server](https://github.com/triton-inference-server/server).

## Features

This framework is meant to satisfy the following requirements:

- **Ease of use** - The framework simplifies the deployment process by handling the Docker containers itself, abstracting away the deployment issue.
- **Containers on demand** - Triton containers scale to zero when idle and wake on the next request, to save resources when possible.
- **Robustness** - Models are deployed using a vanilla NVIDIA Triton Inference Server, offering a powerful and feature-rich platform for every necessity.

## Concept

![architecture](docs/assets/triton-serve.png)

The architecture of Triton Serve is quite straightforward: the overall system is composed of a single management backend, a reconciler and a builder worker, a single reverse proxy/load balancer, and a variable number of Triton container instances.

### Management Backend

The backend represents the main entry point of the tool. This service provides standard a REST API with operations such as:

- CRUD endpoints to manage models (upload, update, delete models and so on)
- CRUD endpoints to manage services (create, update, delete Triton services)

This service aims to simplify the model deployment phase for every user, even without prior knowledge about Docker or best practices for model deployment in general.
The backend is a declarative store: creating a service writes a record and returns, it never touches Docker itself.

### Workers

Two Celery workers act on what the backend records:

- the **reconciler** ticks on a fixed interval, compares what Docker is actually doing against the stored records, and takes the one action that closes the gap
- the **builder** builds and pushes the content-addressed runtime image a service needs before it can start

### Proxy

The "proxy" actually provides several features:

- it acts as the main (and only) entry point for every service in the framework
- it automatically registers the Triton services under a subpath dynamically
- it handles the on-demand provision of these services: a `forwardAuth` middleware asks the backend whether the target service is ready, which records the wake and forwards only once the reconciler has the container up.

### Triton services

In practice, Triton services are nothing more than a vanilla Triton container, programmatically launched and managed by the backend service. These containers are launched in explicit mode so that only the required list of models is loaded on startup.

## Installation

The framework uses `docker-compose`, and it can be run through the provided makefile with a few commands.

```console
make run TARGET=dev|prod [ARGS="-d --build..."]
```

It is also possible to run on cpu-only mode (thus removing the GPU capabilities) by adding the optional `PROFILE` parameter.
For instance:

```console
make run TARGET=dev PROFILE=cpu
```

## Development

The main bulk of code is Python-based. Dependencies are managed with [uv](https://docs.astral.sh/uv/), which also provisions the interpreter.

```bash
git clone https://github.com/links-ads/triton-serve
cd triton-serve
make install
```

`make lint`, `make typecheck` and `make test` are the gates; the test suite runs against a containerised stack rather than in-process.
