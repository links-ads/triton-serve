import pytest

from triton_serve.api.services.reconcile import Action, ObservedState, decide
from triton_serve.database.model import DesiredState, RuntimeStatus

A = ObservedState
D = DesiredState
R = RuntimeStatus


def test_available_running_target1_is_ready():
    d = decide(D.AVAILABLE, A.RUNNING, drifted=False, replica_target=1, attempts=0, max_attempts=3)
    assert (d.action, d.status) == (Action.NONE, R.READY)


def test_available_running_target0_stops_to_idle():
    d = decide(D.AVAILABLE, A.RUNNING, drifted=False, replica_target=0, attempts=0, max_attempts=3)
    assert (d.action, d.status) == (Action.STOP, R.IDLE)


def test_available_absent_target1_recreates_the_outage_cell():
    d = decide(D.AVAILABLE, A.ABSENT, drifted=False, replica_target=1, attempts=0, max_attempts=3)
    assert (d.action, d.status) == (Action.RECREATE, R.WARMING)


def test_available_absent_target0_is_idle_no_action():
    d = decide(D.AVAILABLE, A.ABSENT, drifted=False, replica_target=0, attempts=0, max_attempts=3)
    assert (d.action, d.status) == (Action.NONE, R.IDLE)


def test_available_exited_ok_target1_starts():
    d = decide(D.AVAILABLE, A.EXITED_OK, drifted=False, replica_target=1, attempts=0, max_attempts=3)
    assert (d.action, d.status) == (Action.START, R.WARMING)


def test_available_crashed_within_budget_recovers_and_increments():
    d = decide(D.AVAILABLE, A.CRASHED, drifted=False, replica_target=1, attempts=1, max_attempts=3)
    assert (d.action, d.status, d.increment_attempt) == (Action.RECREATE, R.RECOVERING, True)


def test_available_crashed_budget_exhausted_fails():
    d = decide(D.AVAILABLE, A.CRASHED, drifted=False, replica_target=1, attempts=3, max_attempts=3)
    assert (d.action, d.status) == (Action.MARK_FAILED, R.FAILED)


def test_available_crashed_target0_defers_but_flags_failed_if_spent():
    assert decide(D.AVAILABLE, A.CRASHED, drifted=False, replica_target=0, attempts=0, max_attempts=3).status == R.IDLE
    assert (
        decide(D.AVAILABLE, A.CRASHED, drifted=False, replica_target=0, attempts=3, max_attempts=3).status == R.FAILED
    )


def test_available_image_missing_target1_pulls():
    d = decide(D.AVAILABLE, A.IMAGE_MISSING, drifted=False, replica_target=1, attempts=0, max_attempts=3)
    assert (d.action, d.status) == (Action.PULL, R.WARMING)


def test_available_image_missing_exhausted_fails():
    d = decide(D.AVAILABLE, A.IMAGE_MISSING, drifted=False, replica_target=1, attempts=3, max_attempts=3)
    assert (d.action, d.status) == (Action.MARK_FAILED, R.FAILED)


def test_available_booting_target1_waits_warming():
    d = decide(D.AVAILABLE, A.BOOTING, drifted=False, replica_target=1, attempts=0, max_attempts=3)
    assert (d.action, d.status) == (Action.NONE, R.WARMING)


def test_failed_is_terminal_when_exhausted_across_bringup_facts():
    # a FAILED (budget-exhausted) service must not auto-revive, even once its dead container is
    # removed (-> ABSENT) or a stale exited one lingers (-> EXITED_OK)
    for observed in (A.ABSENT, A.EXITED_OK, A.CRASHED, A.IMAGE_MISSING):
        d = decide(D.AVAILABLE, observed, drifted=False, replica_target=1, attempts=3, max_attempts=3)
        assert (d.action, d.status) == (Action.MARK_FAILED, R.FAILED)


@pytest.mark.parametrize("observed", list(ObservedState))
def test_suspended_stops_live_else_noop(observed):
    d = decide(D.SUSPENDED, observed, drifted=False, replica_target=0, attempts=0, max_attempts=3)
    assert d.status == R.SUSPENDED
    if observed in (A.RUNNING, A.BOOTING):
        assert d.action == Action.STOP
    else:
        assert d.action == Action.NONE


@pytest.mark.parametrize("observed", list(ObservedState))
def test_retired_removes_or_finalizes(observed):
    d = decide(D.RETIRED, observed, drifted=False, replica_target=0, attempts=0, max_attempts=3)
    assert d.status == R.RETIRED
    if observed in (A.RUNNING, A.BOOTING, A.EXITED_OK, A.CRASHED):
        assert d.action == Action.REMOVE
    else:
        assert d.action == Action.FINALIZE


def test_decide_is_total():
    for desired in DesiredState:
        for observed in ObservedState:
            for target in (0, 1):
                assert (
                    decide(desired, observed, drifted=False, replica_target=target, attempts=0, max_attempts=3)
                    is not None
                )


def test_available_image_pending_waits_warming():
    d = decide(D.AVAILABLE, A.IMAGE_PENDING, drifted=False, replica_target=1, attempts=0, max_attempts=3)
    assert (d.action, d.status, d.increment_attempt) == (Action.NONE, R.WARMING, False)


def test_available_image_pending_does_not_spend_the_crash_budget():
    d = decide(D.AVAILABLE, A.IMAGE_PENDING, drifted=False, replica_target=1, attempts=3, max_attempts=3)
    assert (d.action, d.status) == (Action.NONE, R.WARMING)


def test_available_image_failed_is_terminal():
    d = decide(D.AVAILABLE, A.IMAGE_FAILED, drifted=False, replica_target=1, attempts=0, max_attempts=3)
    assert (d.action, d.status) == (Action.MARK_FAILED, R.FAILED)


@pytest.mark.parametrize("observed", [A.RUNNING, A.BOOTING, A.EXITED_OK, A.CRASHED, A.IMAGE_MISSING])
def test_drift_recreates_whatever_the_container_is_doing(observed):
    # one branch covers every liveness fact: a container that no longer matches the row is wrong
    # even while it is healthy, and starting the stopped one (EXITED_OK) is exactly the bug in #128
    d = decide(D.AVAILABLE, observed, drifted=True, replica_target=1, attempts=1, max_attempts=3)
    assert (d.action, d.status) == (Action.RECREATE, R.WARMING)
    assert d.increment_attempt is False
    assert d.reset_attempts is True


def test_edit_revives_a_failed_service():
    d = decide(D.AVAILABLE, A.CRASHED, drifted=True, replica_target=1, attempts=3, max_attempts=3)
    assert (d.action, d.status) == (Action.RECREATE, R.WARMING)
    assert d.reset_attempts is True


def test_drift_is_ignored_while_scaled_to_zero():
    # an edit must never wake a sleeping service; the wake path recreates instead
    d = decide(D.AVAILABLE, A.EXITED_OK, drifted=True, replica_target=0, attempts=0, max_attempts=3)
    assert (d.action, d.status) == (Action.NONE, R.IDLE)


@pytest.mark.parametrize("observed", list(ObservedState))
def test_drift_does_not_change_the_suspended_verdict(observed):
    d = decide(D.SUSPENDED, observed, drifted=True, replica_target=0, attempts=0, max_attempts=3)
    assert d.status == R.SUSPENDED
    assert d.action == (Action.STOP if observed in (A.RUNNING, A.BOOTING) else Action.NONE)


@pytest.mark.parametrize("observed", list(ObservedState))
def test_drift_never_revives_a_retired_service(observed):
    # a tombstoned service still has rows and still fingerprints; it must stay torn down
    d = decide(D.RETIRED, observed, drifted=True, replica_target=0, attempts=0, max_attempts=3)
    assert d.status == R.RETIRED
    expected = Action.REMOVE if observed in (A.RUNNING, A.BOOTING, A.EXITED_OK, A.CRASHED) else Action.FINALIZE
    assert d.action == expected
