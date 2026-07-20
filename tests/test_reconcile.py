import pytest

from triton_serve.api.services.reconcile import Action, ObservedFact, decide
from triton_serve.database.model import DesiredState, RuntimeStatus

A = ObservedFact
D = DesiredState
R = RuntimeStatus


def test_available_running_target1_is_ready():
    d = decide(D.AVAILABLE, A.RUNNING, replica_target=1, attempts=0, max_attempts=3)
    assert (d.action, d.status) == (Action.NONE, R.READY)


def test_available_running_target0_stops_to_idle():
    d = decide(D.AVAILABLE, A.RUNNING, replica_target=0, attempts=0, max_attempts=3)
    assert (d.action, d.status) == (Action.STOP, R.IDLE)


def test_available_absent_target1_recreates_the_outage_cell():
    d = decide(D.AVAILABLE, A.ABSENT, replica_target=1, attempts=0, max_attempts=3)
    assert (d.action, d.status) == (Action.RECREATE, R.WARMING)


def test_available_absent_target0_is_idle_no_action():
    d = decide(D.AVAILABLE, A.ABSENT, replica_target=0, attempts=0, max_attempts=3)
    assert (d.action, d.status) == (Action.NONE, R.IDLE)


def test_available_exited_ok_target1_starts():
    d = decide(D.AVAILABLE, A.EXITED_OK, replica_target=1, attempts=0, max_attempts=3)
    assert (d.action, d.status) == (Action.START, R.WARMING)


def test_available_crashed_within_budget_recovers_and_increments():
    d = decide(D.AVAILABLE, A.CRASHED, replica_target=1, attempts=1, max_attempts=3)
    assert (d.action, d.status, d.increment_attempt) == (Action.RECREATE, R.RECOVERING, True)


def test_available_crashed_budget_exhausted_fails():
    d = decide(D.AVAILABLE, A.CRASHED, replica_target=1, attempts=3, max_attempts=3)
    assert (d.action, d.status) == (Action.MARK_FAILED, R.FAILED)


def test_available_crashed_target0_defers_but_flags_failed_if_spent():
    assert decide(D.AVAILABLE, A.CRASHED, 0, attempts=0, max_attempts=3).status == R.IDLE
    assert decide(D.AVAILABLE, A.CRASHED, 0, attempts=3, max_attempts=3).status == R.FAILED


def test_available_image_missing_target1_pulls():
    d = decide(D.AVAILABLE, A.IMAGE_MISSING, replica_target=1, attempts=0, max_attempts=3)
    assert (d.action, d.status) == (Action.PULL, R.WARMING)


def test_available_image_missing_exhausted_fails():
    d = decide(D.AVAILABLE, A.IMAGE_MISSING, replica_target=1, attempts=3, max_attempts=3)
    assert (d.action, d.status) == (Action.MARK_FAILED, R.FAILED)


def test_available_booting_target1_waits_warming():
    d = decide(D.AVAILABLE, A.BOOTING, replica_target=1, attempts=0, max_attempts=3)
    assert (d.action, d.status) == (Action.NONE, R.WARMING)


@pytest.mark.parametrize("observed", list(ObservedFact))
def test_suspended_stops_live_else_noop(observed):
    d = decide(D.SUSPENDED, observed, replica_target=0, attempts=0, max_attempts=3)
    assert d.status == R.SUSPENDED
    if observed in (A.RUNNING, A.BOOTING):
        assert d.action == Action.STOP
    else:
        assert d.action == Action.NONE


@pytest.mark.parametrize("observed", list(ObservedFact))
def test_retired_removes_or_finalizes(observed):
    d = decide(D.RETIRED, observed, replica_target=0, attempts=0, max_attempts=3)
    assert d.status == R.RETIRED
    if observed in (A.RUNNING, A.BOOTING, A.EXITED_OK, A.CRASHED):
        assert d.action == Action.REMOVE
    else:
        assert d.action == Action.FINALIZE


def test_decide_is_total():
    for desired in DesiredState:
        for observed in ObservedFact:
            for target in (0, 1):
                assert decide(desired, observed, target, attempts=0, max_attempts=3) is not None
