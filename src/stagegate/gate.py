"""The gate: one call site that routes a capability by its stage.

A capability is registered once, with a name, a stage, and a risk tier. From then
on the call site is ordinary Python -- you call the function -- and the gate
decides what that means:

* ``OBSERVE`` records the intended effect and returns without running anything.
* ``SUGGEST`` puts the intended effect to a human and runs only on approval.
* ``ACT`` runs it.

Every path produces exactly one audit record. The stage is resolved per call, not
per import, so promoting a capability is a configuration change rather than a
deploy, and a tripped kill switch takes effect on the very next call.

**Results are explicit.** A gated call returns a :class:`CapabilityResult`, never
a bare value. This is the one piece of ergonomic friction the library insists on,
and it is deliberate: in shadow mode there *is* no return value, and a design that
quietly hands back ``None`` invites callers to treat "nothing happened" as
"succeeded and returned nothing". The type makes you say which you meant.

Example:
    >>> from stagegate import StageGate, Stage, RiskTier
    >>> gate = StageGate()
    >>> @gate.capability("demo.greet", stage=Stage.ACT, risk=RiskTier.LOW)
    ... def greet(name: str) -> str:
    ...     return f"hello {name}"
    >>> greet("ada").unwrap()
    'hello ada'
"""

from __future__ import annotations

import functools
import inspect
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, ParamSpec, TypeVar, cast

from .approval import (
    DEFAULT_TIMEOUT_SECONDS,
    ApprovalHandler,
    ApprovalRequest,
    ApprovalResponse,
)
from .audit import AuditEvent, AuditSink, InMemoryAuditLog, StreamAuditSink
from .correlation import RunContext, _scoped, current_run, new_id
from .errors import ApprovalError, AuditWriteError, ConfigurationError, NotExecuted
from .killswitch import KillSwitch, KillSwitchState
from .policy import StagePolicy
from .redaction import RedactionPolicy, Redactor, _safe_text
from .stages import Decision, Outcome, RiskTier, Stage

__all__ = ["StageGate", "Capability", "CapabilityResult"]


def _first_line(text: str | None) -> str:
    """First non-empty line of a docstring, or ``""``. Never raises on odd input."""
    for line in (text or "").splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


P = ParamSpec("P")
T = TypeVar("T")

Describer = Callable[..., str] | str


@dataclass(frozen=True, slots=True)
class CapabilityResult[T]:
    """What a gated call returns.

    Attributes:
        outcome: What happened. Check this, or use :attr:`executed`.
        event: The audit record written for this call, chain fields included.
        value: The function's return value. Only meaningful when :attr:`executed`.
        error: The exception the function raised, when ``propagate_errors`` is off.

    Example:
        >>> from stagegate import StageGate, Stage
        >>> gate = StageGate()
        >>> @gate.capability("demo.two", stage=Stage.ACT)
        ... def two() -> int:
        ...     return 2
        >>> result = two()
        >>> if result.executed:
        ...     print(result.value)
        2
    """

    outcome: Outcome
    event: AuditEvent
    value: T | None = None
    error: BaseException | None = None
    audit_degraded: bool = False
    """True when the record reached only the fallback sink, not the log of record."""

    @property
    def executed(self) -> bool:
        """Whether the underlying function actually ran."""
        return self.outcome.executed

    @property
    def succeeded(self) -> bool:
        """Whether the function ran and returned without raising."""
        return self.outcome is Outcome.EXECUTED

    @property
    def capability(self) -> str:
        """Name of the capability that was called."""
        return self.event.capability

    @property
    def stage(self) -> Stage:
        """Stage that was in force for this call."""
        return self.event.stage

    @property
    def correlation_id(self) -> str:
        """Correlation id of the run this call belongs to."""
        return self.event.correlation_id

    def __bool__(self) -> bool:
        return self.succeeded

    def unwrap(self) -> T:
        """Return the value, or raise explaining why there is none.

        Raises:
            NotExecuted: nothing ran (shadow mode, denial, timeout, kill switch).
            BaseException: whatever the function raised, if it ran and failed and
                ``propagate_errors`` was off.
        """
        if self.outcome is Outcome.EXECUTED:
            return cast(T, self.value)
        if self.outcome is Outcome.FAILED and self.error is not None:
            raise self.error
        raise NotExecuted(
            f"{self.event.capability} did not execute "
            f"(stage={self.event.stage.value}, outcome={self.outcome.value}"
            + (f", reason={self.event.decision_note}" if self.event.decision_note else "")
            + ")",
            outcome=self.outcome,
            capability=self.event.capability,
        )

    def value_or(self, default: T) -> T:
        """Return the value if the call succeeded, else ``default``."""
        return cast(T, self.value) if self.outcome is Outcome.EXECUTED else default


@dataclass(frozen=True)
class Capability:
    """A registered capability and the per-capability overrides that apply to it."""

    name: str
    func: Callable[..., Any]
    stage: Stage
    """Declared stage. The *effective* stage is resolved per call by the policy."""
    risk: RiskTier
    describe: Describer | None = None
    redact: Redactor | None = None
    approval: ApprovalHandler | None = None
    approval_timeout: float | None = None
    propagate_errors: bool | None = None
    doc: str | None = None
    signature: inspect.Signature | None = field(default=None, compare=False, repr=False)

    def bind(self, args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> dict[str, Any]:
        """Map positional and keyword arguments onto parameter names for logging.

        A signature that will not bind is not an error here: the call itself is
        about to raise a perfectly good ``TypeError``, and this only needs to
        produce something loggable in the meantime.
        """
        if self.signature is not None:
            try:
                bound = self.signature.bind(*args, **dict(kwargs))
                bound.apply_defaults()
                return dict(bound.arguments)
            except TypeError:
                pass
        return {"args": list(args), "kwargs": dict(kwargs)}


class StageGate:
    """Registry and call router for staged capabilities.

    Args:
        audit: Sink for audit records. Defaults to an in-memory log, which is the
            right default for a test and the wrong one for production; point it at
            a :class:`~stagegate.audit.JsonlAuditLog`.
        policy: Where effective stages come from. Defaults to an empty policy, so
            declared stages apply as written.
        kill_switch: Guard consulted before anything executes. Defaults to a
            switch reading ``STAGEGATE_KILL`` and ``$STAGEGATE_KILL_FILE``.
        approval: Default approval handler for ``SUGGEST`` capabilities. Leaving
            this ``None`` is safe, not permissive: a ``SUGGEST`` call with no
            handler has nobody to approve it, so it is refused and audited.
        redaction: Default redaction policy. Defaults to secrets + PII.
        approval_timeout: Default seconds to wait for a decision.
        actor: Identity recorded when a run does not supply one.
        default_stage: Stage used when a capability does not declare one.
        propagate_errors: Whether an exception from inside a capability
            re-raises after the audit record is written. Default ``True``: gating
            a function should not change what happens when it fails. ``False``
            suppresses ordinary exceptions only -- ``KeyboardInterrupt`` and
            ``SystemExit`` always escape.
        strict_audit: When ``True`` (default), the sink must pass a preflight
            check before any execution is allowed. A capability that cannot be
            audited does not run.
        fallback_sink: Where records go if the primary sink fails mid-run.
            Defaults to JSONL on stderr.
        labels: Tags copied onto every event from this gate.

    Example:
        >>> from stagegate import StageGate, Stage, RiskTier, StagePolicy
        >>> gate = StageGate(policy=StagePolicy(max_stage=Stage.OBSERVE))
        >>> @gate.capability("db.drop_table", stage=Stage.ACT, risk=RiskTier.CRITICAL)
        ... def drop_table(table: str) -> None:
        ...     raise AssertionError("never reached under an OBSERVE ceiling")
        >>> drop_table("customers").outcome.value
        'recorded'
    """

    def __init__(
        self,
        *,
        audit: AuditSink | None = None,
        policy: StagePolicy | None = None,
        kill_switch: KillSwitch | None = None,
        approval: ApprovalHandler | None = None,
        redaction: Redactor | None = None,
        approval_timeout: float = DEFAULT_TIMEOUT_SECONDS,
        actor: str | None = None,
        default_stage: Stage = Stage.OBSERVE,
        propagate_errors: bool = True,
        strict_audit: bool = True,
        fallback_sink: AuditSink | None = None,
        labels: Mapping[str, str] | None = None,
    ) -> None:
        self.audit: AuditSink = audit if audit is not None else InMemoryAuditLog()
        self.policy = policy if policy is not None else StagePolicy()
        self.kill_switch = kill_switch if kill_switch is not None else KillSwitch()
        self.approval = approval
        self.redaction: Redactor = redaction if redaction is not None else RedactionPolicy()
        self.approval_timeout = approval_timeout
        self.actor = actor
        self.default_stage = Stage.parse(default_stage)
        self.propagate_errors = propagate_errors
        self.strict_audit = strict_audit
        self.fallback_sink: AuditSink = (
            fallback_sink if fallback_sink is not None else StreamAuditSink()
        )
        self.labels = dict(labels) if labels else None
        self._registry: dict[str, Capability] = {}

    # ---------------------------------------------------------------- registry

    def capability(
        self,
        name: str | None = None,
        *,
        stage: Stage | str | None = None,
        risk: RiskTier | str = RiskTier.MODERATE,
        describe: Describer | None = None,
        redact: Redactor | None = None,
        approval: ApprovalHandler | None = None,
        approval_timeout: float | None = None,
        propagate_errors: bool | None = None,
        replace: bool = False,
    ) -> Callable[[Callable[P, T]], Callable[P, CapabilityResult[T]]]:
        """Register a function as a gated capability.

        Args:
            name: Capability name. Defaults to the function's ``__name__``. This
                name is the policy surface -- overrides and glob patterns match on
                it -- so prefer a namespaced one (``"jira.transition_issue"``) that
                stays stable if the function is renamed.
            stage: Declared stage. Defaults to the gate's ``default_stage``, which
                is ``OBSERVE``: a capability nobody has thought about is in shadow
                mode, not acting.
            risk: Declared blast radius. Recorded, and available to policy
                ceilings.
            describe: How to phrase the intended effect for a human. A format
                string over the argument names (``"Move {issue} to {status}"``) or
                a callable taking the arguments as keywords. It receives the
                *redacted* arguments, never the raw ones.
            redact: Redaction policy for this capability, overriding the gate's.
            approval: Approval handler for this capability, overriding the gate's.
            approval_timeout: Approval timeout for this capability.
            propagate_errors: Override error propagation for this capability.
            replace: Allow re-registering an existing name. Off by default so a
                copy-pasted decorator cannot silently shadow a capability that
                policy already governs.

        Returns:
            A decorator producing a wrapper that returns :class:`CapabilityResult`.

        Raises:
            ConfigurationError: on a duplicate name, or an unparseable stage or
                risk tier.
        """

        def decorator(func: Callable[P, T]) -> Callable[P, CapabilityResult[T]]:
            if inspect.iscoroutinefunction(func):
                raise ConfigurationError(
                    f"{getattr(func, '__name__', func)!r} is a coroutine function; "
                    "StageGate wraps synchronous callables only (see README, Limitations)."
                )
            capability_name = name or getattr(func, "__name__", None) or repr(func)
            try:
                declared = Stage.parse(stage) if stage is not None else self.default_stage
                tier = RiskTier.parse(risk)
            except ValueError as exc:
                raise ConfigurationError(f"{capability_name}: {exc}") from exc

            if capability_name in self._registry and not replace:
                raise ConfigurationError(
                    f"capability {capability_name!r} is already registered; "
                    "pass replace=True if that is intended."
                )

            try:
                signature: inspect.Signature | None = inspect.signature(func)
            except (TypeError, ValueError):  # pragma: no cover - exotic callables
                signature = None

            entry = Capability(
                name=capability_name,
                func=func,
                stage=declared,
                risk=tier,
                describe=describe,
                redact=redact,
                approval=approval,
                approval_timeout=approval_timeout,
                propagate_errors=propagate_errors,
                doc=inspect.getdoc(func),
                signature=signature,
            )
            self._registry[capability_name] = entry

            @functools.wraps(func)
            def wrapper(*args: P.args, **kwargs: P.kwargs) -> CapabilityResult[T]:
                return self._call(entry, args, kwargs)

            wrapper.capability = entry  # type: ignore[attr-defined]
            return wrapper

        return decorator

    def register(
        self,
        func: Callable[..., Any],
        *,
        name: str | None = None,
        stage: Stage | str | None = None,
        risk: RiskTier | str = RiskTier.MODERATE,
        **options: Any,
    ) -> Callable[..., CapabilityResult[Any]]:
        """Register an existing function without decorator syntax.

        For wrapping something you do not own -- a vendor SDK method, a bound
        method on a client object.
        """
        return self.capability(name=name, stage=stage, risk=risk, **options)(func)

    @property
    def capabilities(self) -> Mapping[str, Capability]:
        """Read-only view of the registry."""
        return dict(self._registry)

    def get(self, name: str) -> Capability:
        """Look up a capability by name.

        Raises:
            KeyError: if no such capability is registered.
        """
        try:
            return self._registry[name]
        except KeyError:
            raise KeyError(
                f"no capability named {name!r}; registered: "
                f"{', '.join(sorted(self._registry)) or '(none)'}"
            ) from None

    def invoke(self, name: str, *args: Any, **kwargs: Any) -> CapabilityResult[Any]:
        """Call a capability by name -- the dispatch path for tool-calling agents.

        A model emits a tool name and arguments; this routes it through the same
        gate, policy, kill switch and audit trail as a direct call.

        Raises:
            KeyError: if no such capability is registered. Deliberately not a
                silent no-op: a model naming a tool that does not exist is a
                problem the agent host needs to see and feed back to the model.
        """
        return self._call(self.get(name), args, kwargs)

    def manifest(self) -> list[dict[str, Any]]:
        """Describe every capability and the stage it would run at right now.

        Useful at startup ("here is what this agent can do and how far each
        capability is promoted") and as the header of a dry-run report.
        """
        rows: list[dict[str, Any]] = []
        for entry in sorted(self._registry.values(), key=lambda c: c.name):
            resolution = self.policy.resolve(entry.name, entry.stage, entry.risk)
            rows.append(
                {
                    "capability": entry.name,
                    "declared_stage": entry.stage.value,
                    "effective_stage": resolution.stage.value,
                    "stage_source": resolution.source,
                    "risk_tier": entry.risk.value,
                    "has_approval_handler": bool(entry.approval or self.approval),
                    "summary": _first_line(entry.doc),
                }
            )
        return rows

    # ------------------------------------------------------------------- call

    def _call(
        self, entry: Capability, args: tuple[Any, ...], kwargs: Mapping[str, Any]
    ) -> CapabilityResult[Any]:
        """Route one invocation. Exactly one audit record leaves this method."""
        run = current_run()
        event_id = new_id("evt")
        correlation_id = run.correlation_id if run else new_id("run")
        actor = (run.actor if run and run.actor else None) or self.actor
        labels = self._merge_labels(run)

        redactor = entry.redact or self.redaction
        raw_arguments = entry.bind(args, kwargs)
        arguments, redaction_failed = self._redact(redactor, raw_arguments)
        effect = self._describe(entry, arguments, redactor)

        resolution = self.policy.resolve(entry.name, entry.stage, entry.risk)
        stage = resolution.stage
        declared = entry.stage
        degraded_from: Stage | None = None
        kill_state: KillSwitchState | None = None
        decision = Decision.NOT_REQUIRED
        decision_actor: str | None = None
        notes: list[str] = []
        if resolution.source != "declared":
            notes.append(f"stage from {resolution.source}")
        if redaction_failed:
            notes.append("redaction policy raised; arguments withheld")
        approval_latency_ms: float | None = None
        duration_ms: float | None = None
        outcome = Outcome.RECORDED
        value: Any = None
        error: BaseException | None = None

        # --- guards: anything that could stop execution runs before execution ---
        if stage > Stage.OBSERVE:
            kill_state = self._read_kill_switch()
            if kill_state.tripped:
                degraded_from, stage = stage, Stage.OBSERVE
                outcome = Outcome.BLOCKED
                notes.append(kill_state.reason or "kill switch tripped")

        if stage > Stage.OBSERVE and self.strict_audit:
            audit_problem = self._preflight_audit()
            if audit_problem is not None:
                degraded_from, stage = stage, Stage.OBSERVE
                outcome = Outcome.BLOCKED
                notes.append(f"audit sink unavailable: {audit_problem}")

        # --- routing ---
        if stage is Stage.SUGGEST:
            decision, decision_actor, note, approval_latency_ms = self._seek_approval(
                entry, arguments, effect, correlation_id, event_id, actor, labels, redactor
            )
            if note:
                notes.append(note)
            if decision is not Decision.APPROVED:
                outcome = {
                    Decision.DENIED: Outcome.DENIED,
                    Decision.TIMED_OUT: Outcome.TIMED_OUT,
                    Decision.ERROR: Outcome.ERROR,
                }[decision]

        should_execute = (stage is Stage.ACT) or (
            stage is Stage.SUGGEST and decision is Decision.APPROVED
        )

        if should_execute:
            child = (run or RunContext(correlation_id, actor=actor, labels=labels)).child(event_id)
            started = time.perf_counter()
            try:
                with _scoped(child):
                    value = entry.func(*args, **dict(kwargs))
                outcome = Outcome.EXECUTED
            except BaseException as exc:  # noqa: BLE001 - re-raised below; see _should_reraise
                error = exc
                outcome = Outcome.FAILED
            finally:
                duration_ms = round((time.perf_counter() - started) * 1000, 3)
        elif stage is Stage.OBSERVE and outcome is not Outcome.BLOCKED:
            outcome = Outcome.RECORDED

        event = AuditEvent(
            capability=entry.name,
            stage=stage,
            declared_stage=declared,
            outcome=outcome,
            decision=decision,
            correlation_id=correlation_id,
            event_id=event_id,
            risk_tier=entry.risk,
            arguments=arguments,
            effect=effect,
            actor=actor,
            decision_actor=decision_actor,
            decision_note="; ".join(notes) if notes else None,
            approval_latency_ms=approval_latency_ms,
            duration_ms=duration_ms,
            parent_event_id=run.parent_event_id if run else None,
            depth=run.depth if run else 0,
            error=self._error_record(error, redactor),
            kill_switch=kill_state.to_record() if kill_state else None,
            degraded_from=degraded_from,
            labels=labels,
        )
        written, audit_degraded = self._emit(event)

        if error is not None and self._should_reraise(entry, error):
            raise error

        return CapabilityResult(
            outcome=outcome,
            event=written,
            value=value,
            error=error,
            audit_degraded=audit_degraded,
        )

    # -------------------------------------------------------------- internals

    def _merge_labels(self, run: RunContext | None) -> dict[str, str] | None:
        run_labels = dict(run.labels) if run and run.labels else {}
        merged = {**self.labels, **run_labels} if self.labels else run_labels
        return merged or None

    def _redact(self, redactor: Redactor, raw: Mapping[str, Any]) -> tuple[dict[str, Any], bool]:
        """Apply redaction, withholding everything if the policy itself fails.

        A redactor that raises must not result in raw arguments being logged. The
        record still exists -- you lose the arguments, not the event.
        """
        try:
            result = redactor(raw)
            return {str(k): v for k, v in result.items()}, False
        except Exception:  # noqa: BLE001 - fail closed on the arguments, not the call
            return {"[REDACTION_FAILED]": sorted(str(k) for k in raw)}, True

    def _describe(self, entry: Capability, arguments: Mapping[str, Any], redactor: Redactor) -> str:
        """Phrase the intended effect. Always from redacted arguments, always scrubbed."""
        describe = entry.describe
        text: str
        try:
            if isinstance(describe, str):
                text = describe.format(**arguments)
            elif callable(describe):
                text = str(describe(**arguments))
            else:
                rendered = ", ".join(f"{k}={v!r}" for k, v in arguments.items())
                text = f"{entry.name}({rendered})"
        except Exception as exc:  # noqa: BLE001 - a bad describer must not break a call
            text = f"{entry.name}(<description unavailable: {type(exc).__name__}>)"
        return _safe_text(text, redactor, limit=1000)

    def _read_kill_switch(self) -> KillSwitchState:
        try:
            return self.kill_switch.check()
        except Exception as exc:  # noqa: BLE001 - an unreadable switch is a tripped switch
            return KillSwitchState(
                True, "error", f"kill-switch check raised {type(exc).__name__}: {exc}", time.time()
            )

    def _preflight_audit(self) -> str | None:
        try:
            self.audit.preflight()
        except Exception as exc:  # noqa: BLE001
            return f"{type(exc).__name__}: {exc}"
        return None

    def _seek_approval(
        self,
        entry: Capability,
        arguments: Mapping[str, Any],
        effect: str,
        correlation_id: str,
        event_id: str,
        actor: str | None,
        labels: Mapping[str, str] | None,
        redactor: Redactor,
    ) -> tuple[Decision, str | None, str | None, float | None]:
        """Ask a human. Every failure mode here resolves to "not approved"."""
        handler = entry.approval or self.approval
        if handler is None:
            return (
                Decision.ERROR,
                None,
                f"{entry.name} is at SUGGEST but no approval handler is configured",
                None,
            )

        timeout = (
            entry.approval_timeout if entry.approval_timeout is not None else self.approval_timeout
        )
        request = ApprovalRequest(
            request_id=new_id("apr"),
            capability=entry.name,
            effect=effect,
            arguments=arguments,
            risk_tier=entry.risk,
            stage=Stage.SUGGEST,
            correlation_id=correlation_id,
            event_id=event_id,
            requested_at=time.monotonic(),
            timeout_s=timeout,
            actor=actor,
            labels=labels,
        )
        started = time.perf_counter()
        try:
            response = handler.request_approval(request)
        except TimeoutError as exc:
            latency = round((time.perf_counter() - started) * 1000, 3)
            timed_out = _safe_text(str(exc) or "approval timed out", redactor)
            return Decision.TIMED_OUT, None, timed_out, latency
        except ApprovalError as exc:
            latency = round((time.perf_counter() - started) * 1000, 3)
            failed = _safe_text(f"approval handler failed: {exc}", redactor)
            return Decision.ERROR, None, failed, latency
        except Exception as exc:  # noqa: BLE001 - a handler that breaks cannot approve
            latency = round((time.perf_counter() - started) * 1000, 3)
            return (
                Decision.ERROR,
                None,
                _safe_text(f"approval handler raised {type(exc).__name__}: {exc}", redactor),
                latency,
            )
        latency = round((time.perf_counter() - started) * 1000, 3)

        if not isinstance(response, ApprovalResponse):
            return (
                Decision.ERROR,
                None,
                f"approval handler returned {type(response).__name__}, expected ApprovalResponse",
                latency,
            )
        note = _safe_text(response.note, redactor) if response.note else None
        if response.approved:
            return Decision.APPROVED, response.actor, note, latency
        return Decision.DENIED, response.actor, note, latency

    def _should_reraise(self, entry: Capability, error: BaseException) -> bool:
        """Whether an exception from inside a capability escapes after auditing.

        ``propagate_errors=False`` suppresses *ordinary* failures so the caller can
        inspect ``result.error`` instead. It does not suppress ``KeyboardInterrupt``,
        ``SystemExit`` or any other non-``Exception`` ``BaseException``: those are
        control flow for the process, not a failure of the capability, and
        swallowing them turns Ctrl-C into a hang and ``sys.exit()`` into a no-op.
        """
        if not isinstance(error, Exception):
            return True
        return (
            entry.propagate_errors if entry.propagate_errors is not None else self.propagate_errors
        )

    def _error_record(
        self, error: BaseException | None, redactor: Redactor
    ) -> dict[str, Any] | None:
        if error is None:
            return None
        return {
            "type": type(error).__name__,
            "message": _safe_text(str(error), redactor),
        }

    def _emit(self, event: AuditEvent) -> tuple[AuditEvent, bool]:
        """Write the record, falling back to the secondary sink if the primary fails.

        Returns the chained event and whether only the fallback received it.

        Raises:
            AuditWriteError: if neither sink accepted the record. At that point
                the process has performed an action it cannot account for, and
                the only honest response is to be loud about it.
        """
        try:
            return self.audit.emit(event), False
        except Exception as primary_exc:  # noqa: BLE001
            try:
                return self.fallback_sink.emit(event), True
            except Exception as fallback_exc:  # noqa: BLE001
                raise AuditWriteError(
                    f"audit record for {event.capability} reached no sink "
                    f"(primary: {primary_exc!r}; fallback: {fallback_exc!r})"
                ) from primary_exc
