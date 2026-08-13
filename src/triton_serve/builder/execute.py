import logging
import tempfile
from pathlib import Path

from celery import Task
from docker import DockerClient
from docker.errors import APIError, BuildError

from triton_serve.builder.registry import auth_config, push_auth
from triton_serve.builder.render import write_build_context
from triton_serve.builder.spec import BuildSpec
from triton_serve.config import get_settings
from triton_serve.config.schema import AppSettings
from triton_serve.database import database_manager
from triton_serve.database.model import ImageStatus, ServiceImage, utcnow
from triton_serve.extensions import get_builder_docker_client
from triton_serve.queue import BUILDER_QUEUE, app

LOG = logging.getLogger(__name__)
BUILD_LOG_TAIL = 8000
BUILD_TASK_NAME = "triton_serve.builder.build_image"


def _spec_from_row(image: ServiceImage) -> BuildSpec:
    return BuildSpec(
        base_image=image.base_image or "",
        apt_packages=tuple(image.apt_packages),
        pip_packages=tuple(image.pip_packages),
        pip_index_url=image.pip_index_url,
        pip_extra_index_urls=tuple(image.pip_extra_index_urls),
    )


def _push(client: DockerClient, ref: str, settings: AppSettings) -> None:
    """Pushes a tagged image, turning a streamed error line into the exception docker-py omits."""
    repository, tag = ref.rsplit(":", 1)
    for line in client.images.push(
        repository, tag=tag, auth_config=auth_config(push_auth(settings)), stream=True, decode=True
    ):
        if "errorDetail" in line:
            raise APIError(line["errorDetail"].get("message", "push failed"))


def _mark_failed(image_hash: str, reason: str) -> None:
    with database_manager.session() as db:
        image = db.get(ServiceImage, image_hash)
        if image is not None:
            image.status = ImageStatus.FAILED
            image.build_log = reason[-BUILD_LOG_TAIL:]
            db.commit()
    LOG.error("build %s failed: %s", image_hash[:12], reason[:500])


@app.task(bind=True, name=BUILD_TASK_NAME, queue=BUILDER_QUEUE, max_retries=3)
def build_image(self: Task, image_hash: str) -> None:
    """Builds and pushes the image for a PENDING row, then flips it to READY or FAILED.

    Transient Docker and registry errors are retried with exponential backoff; only once the
    budget is spent does the row go FAILED, which is genuinely terminal because identical inputs
    reproduce identical failures.

    Args:
        image_hash (str): The primary key of the service_images row to build.
    """
    settings = get_settings()
    with database_manager.session() as db:
        image = db.get(ServiceImage, image_hash)
        if image is None or not image.managed or image.status is ImageStatus.READY:
            LOG.info("build %s: nothing to do (missing, unmanaged or already ready)", image_hash[:12])
            return
        image.status = ImageStatus.BUILDING
        db.commit()
        spec = _spec_from_row(image)
        ref = image.image_ref

    client = get_builder_docker_client()
    try:
        with tempfile.TemporaryDirectory() as context:
            write_build_context(spec, Path(context))
            client.images.build(path=context, tag=ref, platform="linux/amd64", rm=True, pull=True)
        _push(client, ref, settings)
    except (BuildError, APIError) as exc:
        if self.request.retries >= self.max_retries:
            _mark_failed(image_hash, str(exc))
            return
        raise self.retry(exc=exc, countdown=30 * 2**self.request.retries) from exc

    with database_manager.session() as db:
        image = db.get(ServiceImage, image_hash)
        if image is not None:
            image.status = ImageStatus.READY
            image.built_at = utcnow()
            image.build_log = None
            db.commit()
    LOG.info("build %s: ready at %s", image_hash[:12], ref)


def enqueue_build(image_hash: str) -> None:
    """Queues a build. Call only after the transaction that created the row has committed."""
    app.send_task(BUILD_TASK_NAME, args=[image_hash], queue=BUILDER_QUEUE)
