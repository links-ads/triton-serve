import enum
from dataclasses import dataclass

from triton_serve.database.model import DesiredState, RuntimeStatus


class ObservedFact(enum.Enum):
    RUNNING = "running"  # container up, health passing
    BOOTING = "booting"  # up, health not yet passing, within boot grace
    EXITED_OK = "exited_ok"  # exited, code 0
    CRASHED = "crashed"  # exited, code != 0
    ABSENT = "absent"  # no container bound to the service (vanished / stale id)
    IMAGE_MISSING = "image_missing"


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


def _available(observed: ObservedFact, target: int, attempts: int, max_attempts: int) -> Decision:
    exhausted = attempts >= max_attempts
    if target == 0:
        # scaled to zero; only surface FAILED if a crash already spent the budget
        if observed in (ObservedFact.RUNNING, ObservedFact.BOOTING):
            return Decision(Action.STOP, RuntimeStatus.IDLE)
        if observed is ObservedFact.CRASHED and exhausted:
            return Decision(Action.NONE, RuntimeStatus.FAILED)
        return Decision(Action.NONE, RuntimeStatus.IDLE)

    # target == 1: drive toward serving
    match observed:
        case ObservedFact.RUNNING:
            return Decision(Action.NONE, RuntimeStatus.READY)
        case ObservedFact.BOOTING:
            return Decision(Action.NONE, RuntimeStatus.WARMING)
        case ObservedFact.EXITED_OK:
            return Decision(Action.START, RuntimeStatus.WARMING)
        case ObservedFact.ABSENT:
            return Decision(Action.RECREATE, RuntimeStatus.WARMING)
        case ObservedFact.CRASHED:
            if exhausted:
                return Decision(Action.MARK_FAILED, RuntimeStatus.FAILED)
            return Decision(Action.RECREATE, RuntimeStatus.RECOVERING, increment_attempt=True)
        case ObservedFact.IMAGE_MISSING:
            if exhausted:
                return Decision(Action.MARK_FAILED, RuntimeStatus.FAILED)
            return Decision(Action.PULL, RuntimeStatus.WARMING, increment_attempt=True)
    raise AssertionError(f"unreachable observed={observed}")  # pragma: no cover


def _suspended(observed: ObservedFact) -> Decision:
    if observed in (ObservedFact.RUNNING, ObservedFact.BOOTING):
        return Decision(Action.STOP, RuntimeStatus.SUSPENDED)
    return Decision(Action.NONE, RuntimeStatus.SUSPENDED)


def _retired(observed: ObservedFact) -> Decision:
    if observed in (ObservedFact.RUNNING, ObservedFact.BOOTING, ObservedFact.EXITED_OK, ObservedFact.CRASHED):
        return Decision(Action.REMOVE, RuntimeStatus.RETIRED)
    return Decision(Action.FINALIZE, RuntimeStatus.RETIRED)


def decide(
    desired: DesiredState,
    observed: ObservedFact,
    replica_target: int,
    attempts: int,
    max_attempts: int,
) -> Decision:
    """Pure reconciliation matrix: maps (intent, observed reality, scale, budget) to one action.

    replica_target is 0/1 and is only ever 1 under AVAILABLE (the autoscaler's decision).
    """
    match desired:
        case DesiredState.AVAILABLE:
            return _available(observed, replica_target, attempts, max_attempts)
        case DesiredState.SUSPENDED:
            return _suspended(observed)
        case DesiredState.RETIRED:
            return _retired(observed)
    raise AssertionError(f"unreachable desired={desired}")  # pragma: no cover
