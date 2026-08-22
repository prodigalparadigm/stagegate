"""Stage, risk tier, decision and outcome vocabularies.

These four enumerations are the stable vocabulary of the audit log. Their string
values are written verbatim into audit records, so treat them as a wire format:
add members freely, never rename or repurpose an existing value.
"""

from __future__ import annotations

from enum import Enum

__all__ = ["Stage", "RiskTier", "Decision", "Outcome"]


class Stage(str, Enum):
    """How far a capability has been promoted toward acting on real systems.

    The three stages are ordered. ``OBSERVE < SUGGEST < ACT``, and that ordering
    is what lets a deployment impose a ceiling (see :class:`~stagegate.policy.StagePolicy`)
    or degrade a capability under a tripped kill switch.
    """

    OBSERVE = "observe"
    """Shadow mode. The intended effect is recorded; nothing executes."""

    SUGGEST = "suggest"
    """The intended effect is put to a human. Execution happens only on approval."""

    ACT = "act"
    """The call executes directly, still fully audited."""

    @property
    def rank(self) -> int:
        """Position in the promotion ladder, 0 (``OBSERVE``) through 2 (``ACT``)."""
        return _STAGE_RANK[self]

    # Stage subclasses str for painless JSON and ``stage == "act"`` comparisons.
    # That inheritance makes ``Stage.ACT < "observe"`` fall back to *lexical* string
    # ordering, which is silently wrong and, in a library about what an agent is
    # allowed to do, dangerous. So ordering against a non-Stage raises rather than
    # returning NotImplemented, which would let the str fallback answer instead.
    def _rank_of(self, other: object) -> int:
        if not isinstance(other, Stage):
            raise TypeError(
                f"cannot order Stage against {type(other).__name__}; "
                f"Stage subclasses str, so this would compare alphabetically. "
                f"Use Stage.parse({other!r}) first."
            )
        return other.rank

    def __lt__(self, other: object) -> bool:
        return self.rank < self._rank_of(other)

    def __le__(self, other: object) -> bool:
        return self.rank <= self._rank_of(other)

    def __gt__(self, other: object) -> bool:
        return self.rank > self._rank_of(other)

    def __ge__(self, other: object) -> bool:
        return self.rank >= self._rank_of(other)

    @classmethod
    def parse(cls, value: str | Stage) -> Stage:
        """Coerce a case-insensitive string to a :class:`Stage`.

        Raises:
            ValueError: if ``value`` names no known stage.
        """
        if isinstance(value, cls):
            return value
        try:
            return cls(str(value).strip().lower())
        except ValueError:
            known = ", ".join(s.value for s in cls)
            raise ValueError(f"unknown stage {value!r}; expected one of: {known}") from None


_STAGE_RANK: dict[Stage, int] = {Stage.OBSERVE: 0, Stage.SUGGEST: 1, Stage.ACT: 2}


class RiskTier(str, Enum):
    """Declared blast radius of a capability.

    StageGate does not enforce anything on the basis of risk tier by itself; it
    records the tier and lets policy key off it. Keeping enforcement out of the
    tier is deliberate, so that the tier stays an honest description of the
    capability rather than a lever people tune to get past a gate.
    """

    LOW = "low"
    """Read-only, or reversible with no external side effect."""

    MODERATE = "moderate"
    """Writes to an internal system; reversible by an operator."""

    HIGH = "high"
    """Writes visible outside the team, or expensive to reverse."""

    CRITICAL = "critical"
    """Irreversible, customer-visible, or financially/legally material."""

    @property
    def rank(self) -> int:
        """Position in the severity ladder, 0 (``LOW``) through 3 (``CRITICAL``)."""
        return _RISK_RANK[self]

    @classmethod
    def parse(cls, value: str | RiskTier) -> RiskTier:
        """Coerce a case-insensitive string to a :class:`RiskTier`."""
        if isinstance(value, cls):
            return value
        try:
            return cls(str(value).strip().lower())
        except ValueError:
            known = ", ".join(r.value for r in cls)
            raise ValueError(f"unknown risk tier {value!r}; expected one of: {known}") from None


_RISK_RANK: dict[RiskTier, int] = {
    RiskTier.LOW: 0,
    RiskTier.MODERATE: 1,
    RiskTier.HIGH: 2,
    RiskTier.CRITICAL: 3,
}


class Decision(str, Enum):
    """What a human (or the absence of one) decided about a call."""

    NOT_REQUIRED = "not_required"
    """No approval was sought: the effective stage was ``OBSERVE`` or ``ACT``."""

    APPROVED = "approved"
    """A human approved the intended effect."""

    DENIED = "denied"
    """A human refused the intended effect."""

    TIMED_OUT = "timed_out"
    """No human answered inside the approval timeout. Treated as a refusal."""

    ERROR = "error"
    """The approval handler itself failed. Treated as a refusal."""


class Outcome(str, Enum):
    """What actually happened to the call."""

    RECORDED = "recorded"
    """Shadow mode: the intended effect was logged and nothing ran."""

    EXECUTED = "executed"
    """The underlying function ran and returned."""

    FAILED = "failed"
    """The underlying function ran and raised."""

    DENIED = "denied"
    """A human refused; nothing ran."""

    TIMED_OUT = "timed_out"
    """Approval timed out; nothing ran."""

    BLOCKED = "blocked"
    """A control-plane guard (kill switch, unavailable audit sink) stopped execution."""

    ERROR = "error"
    """The control plane itself failed; nothing ran."""

    @property
    def executed(self) -> bool:
        """Whether the underlying function was actually invoked."""
        return self in (Outcome.EXECUTED, Outcome.FAILED)
