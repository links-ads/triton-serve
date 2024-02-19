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

RUN cd /usr/local/python/bin && \
    ln -s /usr/local/python/bin/python3 /usr/local/python/bin/python && \
    ln -s /usr/local/python/bin/pip3 /usr/local/python/bin/pip

WORKDIR /app

# copy source, alembic dirs, pyproject.toml and other files
COPY . ./
# install dependencies (no extras)
RUN pip install --no-cache -e .

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