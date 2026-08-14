import enum
from dataclasses import dataclass

from triton_serve.database.model import DesiredState, RuntimeStatus


class ObservedState(enum.Enum):
    RUNNING = "running"  # container up, health passing
    BOOTING = "booting"  # up, health not yet passing, within boot grace
    EXITED_OK = "exited_ok"  # exited, code 0
    CRASHED = "crashed"  # exited, code != 0
    ABSENT = "absent"  # no container bound to the service (vanished / stale id)
    IMAGE_MISSING = "image_missing"
    IMAGE_PENDING = "image_pending"  # build queued or in flight
    IMAGE_FAILED = "image_failed"  # build failed terminally


class Action(enum.Enum):
    NONE = "none"
    START = "start"  # start the existing stopped container (recreate on failure)
    RECREATE = "recreate"  # recreate by service name
    STOP = "stop"  # graceful stop
    REMOVE = "remove"  # stop + remove the container
    PULL = "pull"  # pull the image, then recreate
    FINALIZE = "finalize"  # remove traefik config, release allocation, tombstone
    MARK_FAILED = "mark_failed"


@dataclass(frozen=True)
class Decision:
    action: Action
    status: RuntimeStatus
    increment_attempt: bool = False


def _available(observed: ObservedState, target: int, attempts: int, max_attempts: int) -> Decision:
    exhausted = attempts >= max_attempts
    if target == 0:
        # scaled to zero; only surface FAILED if a crash already spent the budget
        if observed in (ObservedState.RUNNING, ObservedState.BOOTING):
            return Decision(Action.STOP, RuntimeStatus.IDLE)
        if observed is ObservedState.CRASHED and exhausted:
            return Decision(Action.NONE, RuntimeStatus.FAILED)
        return Decision(Action.NONE, RuntimeStatus.IDLE)

    # target == 1: drive toward serving. once the budget is spent every bring-up refuses, so
    # FAILED stays terminal (even if the dead container is later removed) until /retry resets it
    match observed:
        case ObservedState.RUNNING:
            return Decision(Action.NONE, RuntimeStatus.READY)
        case ObservedState.BOOTING:
            return Decision(Action.NONE, RuntimeStatus.WARMING)
        case ObservedState.EXITED_OK:
            if exhausted:
                return Decision(Action.MARK_FAILED, RuntimeStatus.FAILED)
            return Decision(Action.START, RuntimeStatus.WARMING)
        case ObservedState.ABSENT:
            if exhausted:
                return Decision(Action.MARK_FAILED, RuntimeStatus.FAILED)
            return Decision(Action.RECREATE, RuntimeStatus.WARMING)
        case ObservedState.CRASHED:
            if exhausted:
                return Decision(Action.MARK_FAILED, RuntimeStatus.FAILED)
            return Decision(Action.RECREATE, RuntimeStatus.RECOVERING, increment_attempt=True)
        case ObservedState.IMAGE_MISSING:
            if exhausted:
                return Decision(Action.MARK_FAILED, RuntimeStatus.FAILED)
            return Decision(Action.PULL, RuntimeStatus.WARMING, increment_attempt=True)
        case ObservedState.IMAGE_PENDING:
            # a build runs for minutes against a 10s tick; waiting must not spend the crash budget
            return Decision(Action.NONE, RuntimeStatus.WARMING)
        case ObservedState.IMAGE_FAILED:
            return Decision(Action.MARK_FAILED, RuntimeStatus.FAILED)
    raise AssertionError(f"unreachable observed={observed}")  # pragma: no cover


def _suspended(observed: ObservedState) -> Decision:
    if observed in (ObservedState.RUNNING, ObservedState.BOOTING):
        return Decision(Action.STOP, RuntimeStatus.SUSPENDED)
    return Decision(Action.NONE, RuntimeStatus.SUSPENDED)


def _retired(observed: ObservedState) -> Decision:
    if observed in (ObservedState.RUNNING, ObservedState.BOOTING, ObservedState.EXITED_OK, ObservedState.CRASHED):
        return Decision(Action.REMOVE, RuntimeStatus.RETIRED)
    return Decision(Action.FINALIZE, RuntimeStatus.RETIRED)


def decide(
    desired: DesiredState,
    observed: ObservedState,
    replica_target: int,
    attempts: int,
    max_attempts: int,
) -> Decision:
    """Pure reconciliation matrix: maps (intent, observed reality, scale, budget) to one action.

    Args:
        desired (DesiredState): The operator intent recorded on the service.
        observed (ObservedState): The fact observed on the docker daemon.
        replica_target (int): 0 or 1, and only ever 1 under AVAILABLE (the autoscaler's decision).
        attempts (int): Recovery attempts already spent.
        max_attempts (int): The crash budget.

    Returns:
        Decision: The action to apply and the runtime status to project.
    """
    match desired:
        case DesiredState.AVAILABLE:
            return _available(observed, replica_target, attempts, max_attempts)
        case DesiredState.SUSPENDED:
            return _suspended(observed)
        case DesiredState.RETIRED:
            return _retired(observed)
    raise AssertionError(f"unreachable desired={desired}")  # pragma: no cover
