"""Exception hierarchy.

Every exception StageGate raises on its own behalf descends from
:class:`StageGateError`, so an agent host can catch control-plane failures
separately from failures inside the capabilities it wraps.
"""

from __future__ import annotations

__all__ = [
    "StageGateError",
    "ConfigurationError",
    "AuditError",
    "AuditWriteError",
    "AuditLogCorruption",
    "NotExecuted",
    "ApprovalError",
    "RedactionError",
]


class StageGateError(Exception):
    """Base class for every error raised by StageGate itself."""


class ConfigurationError(StageGateError):
    """A gate, policy, or capability was configured in a way that cannot work."""


class AuditError(StageGateError):
    """Base class for audit-log problems."""


class AuditWriteError(AuditError):
    """An audit record could not be durably written."""


class AuditLogCorruption(AuditError):
    """An existing audit log failed its integrity check.

    Attributes:
        path: The log that failed verification, if known.
        seq: Sequence number of the first record that did not verify, if known.
    """

    def __init__(self, message: str, *, path: str | None = None, seq: int | None = None) -> None:
        super().__init__(message)
        self.path = path
        self.seq = seq


class ApprovalError(StageGateError):
    """An approval handler could not produce a decision.

    This is distinct from a denial. A denial is a decision; this is the absence
    of one, and StageGate treats it as a refusal (fail-closed).
    """


class RedactionError(StageGateError):
    """A redaction policy could not be applied, or was used unsafely."""


class NotExecuted(StageGateError):
    """Raised by :meth:`~stagegate.gate.CapabilityResult.unwrap` when nothing ran.

    Attributes:
        outcome: The :class:`~stagegate.stages.Outcome` that explains why.
        capability: Name of the capability that did not execute.
    """

    def __init__(
        self, message: str, *, outcome: object = None, capability: str | None = None
    ) -> None:
        super().__init__(message)
        self.outcome = outcome
        self.capability = capability
