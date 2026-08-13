import logging
from itertools import chain

from sqlalchemy.orm import Session

from triton_serve.builder.registry import image_ref
from triton_serve.builder.spec import BuildSpec, make_build_spec
from triton_serve.config.schema import AppSettings
from triton_serve.database.model import ImageStatus, Model, Service, ServiceImage, utcnow

LOG = logging.getLogger(__name__)


def service_build_spec(service: Service, settings: AppSettings) -> BuildSpec:
    """Builds the spec for a service from its own base image and its models' dependency union.

    Args:
        service (Service): The service, with its models loaded.
        settings (AppSettings): The application settings.

    Returns:
        BuildSpec: The normalized spec.

    Raises:
        ValueError: If a stored dependency fails validation.
    """
    return make_build_spec(
        base_image=service.service_image,
        apt_packages=chain.from_iterable(model.system_dependencies or [] for model in service.models),
        pip_packages=chain.from_iterable(model.dependencies or [] for model in service.models),
        allowed_index_hosts=settings.pip_index_allowed_hosts,
    )


def resolve_service_image(db: Session, service: Service, settings: AppSettings) -> str | None:
    """Points a service at the image row for its current spec, inserting the row if it is new.

    A spec with no packages resolves to an unmanaged READY row whose ref is the base image itself,
    so the reconciler's existing IMAGE_MISSING -> PULL path handles it unchanged. That is also what
    makes a digest-pinned image work with no build and no extra endpoint.

    Args:
        db (Session): The database session.
        service (Service): The service to resolve.
        settings (AppSettings): The application settings.

    Returns:
        str | None: The image hash the caller must enqueue a build for after committing, or None.

    Raises:
        ValueError: If the service's dependency set fails validation.
    """
    spec = service_build_spec(service, settings)
    image = db.get(ServiceImage, spec.image_hash)
    if image is not None:
        service.image_hash = image.image_hash
        return None

    empty = spec.is_empty
    image = ServiceImage(
        image_hash=spec.image_hash,
        image_ref=spec.base_image if empty else image_ref(settings, spec.image_hash),
        status=ImageStatus.READY if empty else ImageStatus.PENDING,
        managed=not empty,
        base_image=spec.base_image,
        apt_packages=list(spec.apt_packages),
        pip_packages=list(spec.pip_packages),
        pip_index_url=spec.pip_index_url,
        pip_extra_index_urls=list(spec.pip_extra_index_urls),
        built_at=utcnow() if empty else None,
    )
    db.add(image)
    db.flush()
    service.image_hash = image.image_hash
    return None if empty else image.image_hash


def services_using_models(db: Session, models: list[Model]) -> list[Service]:
    """Every live service serving any of the given models."""
    model_ids = {model.model_id for model in models}
    if not model_ids:
        return []
    return (
        db.query(Service)
        .join(Service.models)
        .filter(Model.model_id.in_(model_ids), Service.deleted_at.is_(None))
        .distinct()
        .all()
    )
