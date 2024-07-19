ARG TARGET=dev
ARG NVIDIA_DRIVER_VERSION=525-5.15.0-94-generic
ARG UBUNTU_VERSION=22.04
ARG PYTHON_VERSION=3.10.12

# Stage 1: Build Python
FROM nvcr.io/nvidia/driver:${NVIDIA_DRIVER_VERSION}-ubuntu${UBUNTU_VERSION} AS python-build
ARG PYTHON_VERSION

RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    build-essential \
    libffi-dev \
    libssl-dev \
    zlib1g-dev \
    liblzma-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev

RUN wget https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tgz && \
    tar -xzf Python-${PYTHON_VERSION}.tgz && \
    cd Python-${PYTHON_VERSION} && \
    ./configure \
        --prefix /usr/local/python \
        --enable-shared \
        --enable-optimizations && \
    make install -j 4

# Stage 2: Build the final image
FROM nvcr.io/nvidia/driver:${NVIDIA_DRIVER_VERSION}-ubuntu${UBUNTU_VERSION} AS base
COPY --from=python-build /usr/local/python /usr/local/python

LABEL author="LINKS Foundation"
LABEL maintainer="LINKS Foundation"
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/usr/local/python/bin:${PATH}" \
    LD_LIBRARY_PATH="/usr/local/python/lib:${LD_LIBRARY_PATH}" \
    PYTHONPATH="/usr/local/python/bin:${PYTHONPATH}"

# Soft link python3 and pip3 to python and pip
RUN cd /usr/local/python/bin && \
    ln -s /usr/local/python/bin/python3 /usr/local/python/bin/python && \
    ln -s /usr/local/python/bin/pip3 /usr/local/python/bin/pip

# Install git and SSH keys for private repositories
COPY .ssh /root/.ssh
RUN apt-get update && apt-get install -y --no-install-recommends \
    git git-lfs openssh-client && \
    git lfs install && \
    chmod 600 /root/.ssh/id_rsa && \
    ssh-keyscan github.com >> /root/.ssh/known_hosts && \
    ssh-keyscan hf.co >> /root/.ssh/known_hosts

# Change the working directory
WORKDIR /app

# copy requirements and install runtime dependencies
COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# copy source, alembic dirs, pyproject.toml and other files
COPY . ./
# install dependencies: don't use cache to save space,
# don't install dependencies since they are already installed (this saves build time)
RUN pip install --no-cache --no-deps -e .

# copy the entrypoint script
COPY docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh

# Stage 3: Add additional dependencies for development, production and testing
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