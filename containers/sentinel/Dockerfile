ARG TARGET=dev

FROM python:3.10-slim-bullseye AS base
LABEL author="LINKS Foundation"
LABEL maintainer="LINKS Foundation"
ARG TARGET

WORKDIR /app

# copy source, alembic dirs, pyproject.toml and other files
COPY ./containers/sentinel/requirements.txt sentinel.txt
RUN pip install  --no-cache-dir -r sentinel.txt


# copy source, alembic dirs, pyproject.toml and other files
COPY . ./
# install dependencies: don't use cache to save space,
# don't install dependencies since they are already installed (this saves build time)
RUN pip install --no-cache --no-deps -e .

COPY docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh

ENTRYPOINT [ "/docker-entrypoint.sh" ]