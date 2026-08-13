from itertools import chain

from sqlalchemy.orm import Session

from triton_serve.builder.registry import image_ref
from triton_serve.builder.spec import BuildSpec, make_build_spec
from triton_serve.config.schema import AppSettings
from triton_serve.database.model import ImageStatus, Model, Service, ServiceImage, utcnow


def pip_dependencies(service: Service) -> list[str]:
    """The pip requirement union across a service's models."""
    return sorted(set(chain.from_iterable(model.dependencies or [] for model in service.models)))


def system_dependencies(service: Service) -> list[str]:
    """The apt package union across a service's models."""
    return sorted(set(chain.from_iterable(model.system_dependencies or [] for model in service.models)))


def service_build_spec(service: Service) -> BuildSpec:
    """Builds the spec for a service from its own base image and its models' dependency union.

    Args:
        service (Service): The service, with its models loaded.

    Returns:
        BuildSpec: The normalized spec.

    Raises:
        ValueError: If a stored dependency fails validation.
    """
    return make_build_spec(
        base_image=service.service_image,
        apt_packages=system_dependencies(service),
        pip_packages=pip_dependencies(service),
    )


def image_from_spec(spec: BuildSpec, settings: AppSettings) -> ServiceImage:
    """The image row a spec resolves to.

    An empty spec is unmanaged and already ready at its base image: there is nothing to build, so
    the reconciler's existing IMAGE_MISSING -> PULL path handles it unchanged.
    """
    empty = spec.is_empty
    return ServiceImage(
        image_hash=spec.image_hash,
        image_ref=spec.base_image if empty else image_ref(settings, spec.image_hash),
        status=ImageStatus.READY if empty else ImageStatus.PENDING,
        managed=not empty,
        base_image=spec.base_image,
        apt_packages=list(spec.apt_packages),
        pip_packages=list(spec.pip_packages),
        built_at=utcnow() if empty else None,
    )


def resolve_service_image(db: Session, service: Service, settings: AppSettings) -> str | None:
    """Points a service at the image row for its current spec, inserting the row if it is new.

    A spec with no packages needs no build, which is also what makes a digest-pinned image work
    with no extra endpoint.

    Args:
        db (Session): The database session.
        service (Service): The service to resolve.
        settings (AppSettings): The application settings.

    Returns:
        str | None: The image hash the caller must enqueue a build for after committing, or None.

    Raises:
        ValueError: If the service's dependency set fails validation.
    """
    spec = service_build_spec(service)
    image = db.get(ServiceImage, spec.image_hash)
    if image is not None:
        service.image_hash = image.image_hash
        return None

    image = image_from_spec(spec, settings)
    db.add(image)
    db.flush()
    service.image_hash = image.image_hash
    return image.image_hash if image.managed else None


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
