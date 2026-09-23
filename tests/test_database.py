from triton_serve.database import database_manager

_TEST_LOCK_KEY = 0x7213_7E57


def test_a_free_lock_is_acquired_and_released(test_connection):
    with database_manager.advisory_lock(_TEST_LOCK_KEY) as acquired:
        assert acquired is True
    # released on exit, so the next pass takes it rather than skipping forever
    with database_manager.advisory_lock(_TEST_LOCK_KEY) as again:
        assert again is True


def test_a_held_lock_is_refused_to_the_next_caller(test_connection):
    """Single-flight: the second pass must learn it lost, not block until the first finishes."""
    with database_manager.advisory_lock(_TEST_LOCK_KEY) as first:
        assert first is True
        with database_manager.advisory_lock(_TEST_LOCK_KEY) as second:
            assert second is False
