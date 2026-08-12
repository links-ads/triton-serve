from triton_serve.api.services.domain import docker_healthcheck


def test_no_healthcheck_is_none():
    assert docker_healthcheck(None) is None


def test_empty_healthcheck_is_none():
    assert docker_healthcheck({}) is None


def test_seconds_are_converted_to_nanoseconds():
    stored = {
        "test": ["CMD", "curl", "-f", "http://localhost:8000/v2/health/ready"],
        "interval": 10,
        "timeout": 5,
        "retries": 3,
        "start_period": 60,
    }
    assert docker_healthcheck(stored) == {
        "Test": ["CMD", "curl", "-f", "http://localhost:8000/v2/health/ready"],
        "Interval": 10_000_000_000,
        "Timeout": 5_000_000_000,
        "Retries": 3,
        "StartPeriod": 60_000_000_000,
    }
