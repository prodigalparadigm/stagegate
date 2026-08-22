"""StageGate: staged, auditable rollout of agent capabilities.

An agent capability should not go from "someone wrote it" to "it acts on
production systems" in one step. Register each capability with a stage, and the
stage governs what actually happens when the agent invokes it:

* ``OBSERVE`` -- record the intended effect, execute nothing.
* ``SUGGEST`` -- put the intended effect to a human, execute only on approval.
* ``ACT`` -- execute, still fully audited.

Promotion between stages is explicit configuration, never automatic.

Example:
    >>> from stagegate import StageGate, Stage, RiskTier, agent_run
    >>> gate = StageGate()
    >>> @gate.capability("tickets.close", stage=Stage.OBSERVE, risk=RiskTier.MODERATE,
    ...                  describe="Close ticket {ticket_id}")
    ... def close_ticket(ticket_id: str) -> str:
    ...     raise AssertionError("shadow mode: never called")
    >>> with agent_run(actor="triage-bot"):
    ...     result = close_ticket("T-4471")
    >>> result.executed, result.event.effect
    (False, 'Close ticket T-4471')
"""

from __future__ import annotations

from .approval import (
    DEFAULT_TIMEOUT_SECONDS,
    ApprovalHandler,
    ApprovalRequest,
    ApprovalResponse,
    CLIApprovalHandler,
    PendingApproval,
    QueueApprovalHandler,
    StaticApprovalHandler,
)
from .audit import (
    SCHEMA,
    AuditEvent,
    AuditSink,
    ChainVerification,
    InMemoryAuditLog,
    JsonlAuditLog,
    MultiSink,
    StreamAuditSink,
    read_events,
    verify_chain,
)
from .correlation import RunContext, agent_run, current_correlation_id, current_run, new_id
from .errors import (
    ApprovalError,
    AuditError,
    AuditLogCorruption,
    AuditWriteError,
    ConfigurationError,
    NotExecuted,
    RedactionError,
    StageGateError,
)
from .gate import Capability, CapabilityResult, StageGate
from .killswitch import KillSwitch, KillSwitchState
from .policy import StagePolicy, StageResolution
from .redaction import (
    DEFAULT_PII_KEYS,
    DEFAULT_SECRET_KEYS,
    REDACTED,
    RedactionPolicy,
    Redactor,
)
from .report import CapabilityReport, DryRunReport, build_report, report_from_log
from .stages import Decision, Outcome, RiskTier, Stage

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # core
    "StageGate",
    "Capability",
    "CapabilityResult",
    "Stage",
    "RiskTier",
    "Decision",
    "Outcome",
    # policy
    "StagePolicy",
    "StageResolution",
    # correlation
    "agent_run",
    "current_run",
    "current_correlation_id",
    "RunContext",
    "new_id",
    # audit
    "SCHEMA",
    "AuditEvent",
    "AuditSink",
    "JsonlAuditLog",
    "InMemoryAuditLog",
    "StreamAuditSink",
    "MultiSink",
    "ChainVerification",
    "verify_chain",
    "read_events",
    # kill switch
    "KillSwitch",
    "KillSwitchState",
    # approvals
    "ApprovalHandler",
    "ApprovalRequest",
    "ApprovalResponse",
    "CLIApprovalHandler",
    "QueueApprovalHandler",
    "StaticApprovalHandler",
    "PendingApproval",
    "DEFAULT_TIMEOUT_SECONDS",
    # redaction
    "Redactor",
    "RedactionPolicy",
    "REDACTED",
    "DEFAULT_SECRET_KEYS",
    "DEFAULT_PII_KEYS",
    # reporting
    "DryRunReport",
    "CapabilityReport",
    "build_report",
    "report_from_log",
    # errors
    "StageGateError",
    "ConfigurationError",
    "AuditError",
    "AuditWriteError",
    "AuditLogCorruption",
    "ApprovalError",
    "RedactionError",
    "NotExecuted",
]
