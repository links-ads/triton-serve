ARG TARGET=dev

FROM python:3.10-slim-bullseye AS base
LABEL author="LINKS Foundation"
LABEL maintainer="LINKS Foundation"
ARG TARGET

# install additional packages such as gcc
RUN apt-get update && apt-get install -y gcc \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY src/ src/
COPY pyproject.toml .
# install dependencies (no extras)
RUN pip install -e .

FROM base AS branch-dev
RUN echo "Building development branch"
RUN pip install -e .[dev]

FROM base AS branch-prod
RUN echo "Building production branch"

FROM base AS branch-test
RUN echo "Building testing branch"
COPY tests/ tests/
RUN pip install -e .[test]

FROM branch-${TARGET}

EXPOSE 5000