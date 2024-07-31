ARG TARGET=dev

FROM python:3.10-slim-bullseye AS base
LABEL author="LINKS Foundation"
LABEL maintainer="LINKS Foundation"
ARG TARGET

WORKDIR /app

# copy source, alembic dirs, pyproject.toml and other files
COPY ./containers/celery/requirements.txt ./containers/celery/requirements.txt
# install runtime dependencies
RUN pip install -r ./containers/celery/requirements.txt

COPY docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh


COPY . ./
WORKDIR /app/src

ENTRYPOINT [ "/docker-entrypoint.sh" ]