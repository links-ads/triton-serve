ARG TARGET=dev

FROM python:3.10-slim-bullseye AS base
LABEL author="LINKS Foundation"
LABEL maintainer="LINKS Foundation"
ARG TARGET

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# install additional packages such as gcc
RUN apt-get update && apt-get install -y gcc \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# copy source, alembic dirs,pyproject.toml and other files
COPY . ./
# install dependencies (no extras)
RUN pip install -e .

# copy the entrypoint script
COPY docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh

FROM base AS branch-dev
RUN echo "Building development branch"
RUN pip install -e .[dev]

FROM base AS branch-prod
RUN echo "Building production branch"

FROM base AS branch-test
RUN echo "Building testing branch"
RUN pip install -e .[test]
COPY tests/ tests/

FROM branch-${TARGET}

EXPOSE 5000
ENTRYPOINT [ "/docker-entrypoint.sh" ]