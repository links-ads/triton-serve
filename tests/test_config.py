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
    assert settings.image_build_timeout < settings.broker_visibility_timeout < settings.build_stale_after
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


def test_the_hard_limit_sits_under_the_visibility_timeout():
    """What makes a second builder replica safe: the attempt is killed before the broker gives up on
    it, so a redelivered message can never find the first attempt still running."""
    settings = _settings(image_build_timeout=1800, image_build_stale_after=3600)
    assert settings.image_build_timeout < settings.image_build_hard_limit < settings.broker_visibility_timeout


def test_a_gap_too_narrow_for_the_hard_limit_is_rejected():
    """1800/1900 leaves the visibility timeout at 1850 and the hard limit at 1860: the broker would
    redeliver ten seconds before the first attempt is stopped."""
    with pytest.raises(ValidationError, match="image_build_stale_after"):
        _settings(image_build_timeout=1800, image_build_stale_after=1900)


def test_the_narrowest_accepted_gap_keeps_the_hard_limit_below_the_window():
    """Pins where the check falls for a threshold set by hand. A 1800s timeout puts the hard limit
    at 1980, and the window only clears it from 2162 up, which is why the error names no number."""
    with pytest.raises(ValidationError, match="image_build_stale_after"):
        _settings(image_build_timeout=1800, image_build_stale_after=2161)

    settings = _settings(image_build_timeout=1800, image_build_stale_after=2162)
    assert settings.image_build_hard_limit < settings.broker_visibility_timeout


def test_the_stale_threshold_follows_the_timeout_when_it_is_not_set():
    """None of this should have to be set by hand: overriding the timeout alone has to leave a band
    that still validates, which it cannot while the threshold holds an unrelated default."""
    settings = _settings(image_build_timeout=700)

    assert settings.build_stale_after == 1400
    assert settings.image_build_hard_limit < settings.broker_visibility_timeout < settings.build_stale_after


def test_an_explicit_stale_threshold_still_wins():
    """It stays tunable, it just stops being something you must keep in sync by hand."""
    settings = _settings(image_build_timeout=600, image_build_stale_after=2000)

    assert settings.build_stale_after == 2000


def test_azure_storage_without_an_account_is_rejected():
    """The failure belongs at startup, not at the first upload hours later."""
    with pytest.raises(ValidationError, match="azure_storage_account"):
        _settings(storage_type="azure")


def test_azure_storage_without_a_key_is_rejected():
    with pytest.raises(ValidationError, match="azure_storage_key"):
        _settings(storage_type="azure", azure_storage_account="adsmodelrepository")


def test_a_configured_azure_account_validates():
    settings = _settings(
        storage_type="azure",
        azure_storage_account="adsmodelrepository",
        azure_storage_key="deadbeef",
    )
    assert settings.azure_storage_container == "model-repository"
    assert settings.azure_storage_key.get_secret_value() == "deadbeef"


def test_local_storage_needs_no_azure_settings():
    assert _settings().azure_storage_account == ""


def test_the_account_key_is_not_in_the_repr():
    """It reaches every worker's environment; it must not also reach every log line."""
    settings = _settings(
        storage_type="azure",
        azure_storage_account="adsmodelrepository",
        azure_storage_key="deadbeef",
    )
    assert "deadbeef" not in repr(settings)
