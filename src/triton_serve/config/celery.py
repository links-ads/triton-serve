from typing import ClassVar

from triton_serve.config import get_settings

settings = get_settings()


class Config:
    timezone = "UTC"
    broker_url = settings.celery_broker_url
    broker_connection_retry_on_startup = True
    broker_transport_options: ClassVar[dict[str, int]] = {"visibility_timeout": settings.broker_visibility_timeout}
    # with acks_late, a prefetched build is already unacked, so its visibility timer runs while it
    # waits behind the build in progress. one at a time keeps the timer aligned with actual work
    worker_prefetch_multiplier = 1
