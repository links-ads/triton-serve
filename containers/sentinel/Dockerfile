ARG TARGET=dev
ARG PYTHON_VERSION=3.12.12
ARG UV_VERSION=0.10.7

FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS base

FROM python:${PYTHON_VERSION}-slim-trixie
COPY --from=base /uv /uvx /bin/

WORKDIR /app
COPY . ./
RUN uv sync --locked

COPY docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh

ENTRYPOINT [ "/docker-entrypoint.sh" ]