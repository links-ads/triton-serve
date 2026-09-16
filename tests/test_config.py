import pytest
from pydantic import ValidationError

from triton_serve.config.schema import AppSettings

REQUIRED = {
    "database_host": "database",
    "database_user": "postgres",
    "database_pass": "postgres",
    "backend_host": "backend",
    "backend_port": 5000,
}


def _settings(**overrides) -> AppSettings:
    return AppSettings(**{**REQUIRED, **overrides})  # type: ignore[arg-type]


def test_broker_url_points_at_redis():
    settings = _settings(redis_host="redis", redis_port=6379)
    assert settings.celery_broker_url == "redis://redis:6379/0"


def test_visibility_timeout_sits_inside_the_safe_band():
    """Above one build attempt so a live build is never handed to a second worker, below the
    reaper's threshold so a lost attempt is redelivered before the row is failed."""
    settings = _settings(image_build_timeout=1800, image_build_stale_after=3600)
    assert settings.image_build_timeout < settings.broker_visibility_timeout < settings.image_build_stale_after
    assert settings.broker_visibility_timeout == 2700


def test_visibility_timeout_tracks_overridden_bounds():
    settings = _settings(image_build_timeout=600, image_build_stale_after=1000)
    assert 600 < settings.broker_visibility_timeout < 1000
    assert settings.broker_visibility_timeout == 800


def test_inverted_build_bounds_are_rejected():
    with pytest.raises(ValidationError, match="image_build_timeout"):
        _settings(image_build_timeout=3600, image_build_stale_after=1800)


def test_celery_config_pins_the_visibility_timeout():
    from triton_serve.config import get_settings
    from triton_serve.config.celery import Config

    assert Config.broker_url.startswith("redis://")
    assert Config.broker_transport_options["visibility_timeout"] == get_settings().broker_visibility_timeout
    assert Config.worker_prefetch_multiplier == 1
