from celery.schedules import timedelta

from triton_serve.config import get_settings

settings = get_settings()

# broker url for celery
broker_url = settings.celery_broker_url

# schedule for the celery tasks
beat_schedule = {
    "check-containers": {
        "task": "triton_serve.tasks.check_and_update_container_task",
        "schedule": timedelta(seconds=settings.sentinel_poll_interval_s),
    },
}

# retry to connect to the broker on startup.
# Not really needed since depends_on is specified in the docker-compose file.
# It silences a warning.
broker_connection_retry_on_startup = True

# timezone
timezone = "UTC"
