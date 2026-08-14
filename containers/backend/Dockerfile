ARG TARGET=dev
ARG NVIDIA_DRIVER_VERSION=525-5.15.0-94-generic
ARG UBUNTU_VERSION=24.04
ARG PYTHON_VERSION=3.14.3
ARG UV_VERSION=0.10.7

# Stage 1: Build Python
FROM nvcr.io/nvidia/driver:${NVIDIA_DRIVER_VERSION}-ubuntu${UBUNTU_VERSION} AS base
ARG PYTHON_VERSION
ARG UV_VERSION

LABEL author="LINKS Foundation"
LABEL maintainer="LINKS Foundation"
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_NO_DEV=1

# The installer requires curl (and certificates) to download the release archive
RUN apt update && apt install -y --no-install-recommends curl ca-certificates

# Install uv and add binaries to the path
ADD https://astral.sh/uv/${UV_VERSION}/install.sh /uv-installer.sh
RUN sh /uv-installer.sh && rm /uv-installer.sh
ENV PATH="/root/.local/bin/:$PATH"
# install a specific python version
RUN uv python install ${PYTHON_VERSION} && uv python pin ${PYTHON_VERSION}

# Install git and SSH keys for private repositories.
# Trust github + hugging face via a static, version-controlled known_hosts instead of
# ssh-keyscan: keyscan is non-deterministic and fails the build when a host's SSH
# handshake is flaky or unreachable (hf.co in particular). Refresh the file if a
# provider rotates its host keys.
COPY .ssh /root/.ssh
COPY containers/backend/known_hosts /root/.ssh/known_hosts
RUN apt-get update && apt-get install -y --no-install-recommends \
    git git-lfs openssh-client && \
    git lfs install && \
    chmod 600 /root/.ssh/id_rsa && \
    chmod 644 /root/.ssh/known_hosts

# Change the working directory
WORKDIR /app
# copy source, alembic dirs, pyproject.toml and other files
COPY . ./
RUN uv sync --locked

# copy the entrypoint script
COPY docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh

# Stage 3: Add additional dependencies for development, production and testing
FROM base AS branch-dev
RUN echo "Building development branch"
RUN uv sync --group dev

FROM base AS branch-prod
RUN echo "Building production branch"

FROM base AS branch-test
RUN echo "Building testing branch"
RUN uv sync --group test
COPY tests/ tests/

FROM branch-${TARGET}

EXPOSE 5000
ENTRYPOINT [ "/docker-entrypoint.sh" ]
