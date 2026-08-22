"""Stage routing: what actually happens at each stage, and what does not."""

from __future__ import annotations

import re

import pytest

from stagegate import (
    ConfigurationError,
    Decision,
    NotExecuted,
    Outcome,
    RiskTier,
    Stage,
    StageGate,
    StagePolicy,
    StaticApprovalHandler,
    agent_run,
)


def build(gate: StageGate, stage: Stage, calls: list, **options):
    @gate.capability("demo.act", stage=stage, describe="do {target}", **options)
    def do(target: str) -> str:
        calls.append(target)
        return f"did {target}"

    return do


# ------------------------------------------------------------------- OBSERVE


def test_observe_records_the_intent_and_executes_nothing(gate: StageGate, sink, calls) -> None:
    do = build(gate, Stage.OBSERVE, calls)
    result = do("widget-1")

    assert calls == [], "OBSERVE must not invoke the underlying function"
    assert result.outcome is Outcome.RECORDED
    assert result.executed is False
    assert result.value is None
    assert len(sink.events) == 1
    assert sink.events[0].effect == "do widget-1"
    assert sink.events[0].stage is Stage.OBSERVE
    assert sink.events[0].decision is Decision.NOT_REQUIRED
    assert sink.events[0].duration_ms is None


def test_observe_records_full_arguments_not_just_the_name(gate: StageGate, calls) -> None:
    @gate.capability("demo.full", stage=Stage.OBSERVE)
    def send(to: str, subject: str, retries: int = 3) -> None:
        calls.append(to)

    send("ops@example.com", subject="deploy failed")
    arguments = gate.audit.events[0].arguments
    assert arguments["subject"] == "deploy failed"
    assert arguments["retries"] == 3, "defaults are recorded, not just what was passed"


def test_unwrap_on_a_shadow_call_explains_itself(gate: StageGate, calls) -> None:
    do = build(gate, Stage.OBSERVE, calls)
    result = do("widget-1")
    with pytest.raises(NotExecuted, match="did not execute"):
        result.unwrap()
    assert result.value_or("fallback") == "fallback"
    assert bool(result) is False


# ----------------------------------------------------------------------- ACT


def test_act_executes_and_records(gate: StageGate, sink, calls) -> None:
    do = build(gate, Stage.ACT, calls)
    result = do("widget-1")

    assert calls == ["widget-1"]
    assert result.outcome is Outcome.EXECUTED
    assert result.unwrap() == "did widget-1"
    assert bool(result) is True
    assert sink.events[0].decision is Decision.NOT_REQUIRED
    assert sink.events[0].duration_ms is not None and sink.events[0].duration_ms >= 0


def test_a_failing_capability_is_audited_then_re_raised(gate: StageGate, sink) -> None:
    @gate.capability("demo.boom", stage=Stage.ACT)
    def boom() -> None:
        raise RuntimeError("upstream 503")

    with pytest.raises(RuntimeError, match="upstream 503"):
        boom()

    assert len(sink.events) == 1, "the record is written before the exception escapes"
    assert sink.events[0].outcome is Outcome.FAILED
    assert sink.events[0].error == {"type": "RuntimeError", "message": "upstream 503"}
    assert sink.events[0].duration_ms is not None


def test_error_propagation_can_be_turned_off_per_capability(gate: StageGate) -> None:
    @gate.capability("demo.quiet", stage=Stage.ACT, propagate_errors=False)
    def boom() -> None:
        raise RuntimeError("upstream 503")

    result = boom()
    assert result.outcome is Outcome.FAILED
    assert isinstance(result.error, RuntimeError)
    with pytest.raises(RuntimeError, match="upstream 503"):
        result.unwrap()


# ------------------------------------------------------------------- SUGGEST


def test_suggest_executes_only_after_approval(sink, calls) -> None:
    handler = StaticApprovalHandler(True, actor="ops@example.com", note="looks fine")
    gate = StageGate(audit=sink, approval=handler, approval_timeout=1.0)
    do = build(gate, Stage.SUGGEST, calls)

    result = do("widget-1")

    assert calls == ["widget-1"]
    assert result.outcome is Outcome.EXECUTED
    assert sink.events[0].decision is Decision.APPROVED
    assert sink.events[0].decision_actor == "ops@example.com"
    assert sink.events[0].decision_note == "looks fine"
    assert sink.events[0].approval_latency_ms is not None
    assert handler.requests[0].effect == "do widget-1"


def test_suggest_denied_executes_nothing(sink, calls) -> None:
    gate = StageGate(audit=sink, approval=StaticApprovalHandler(False, actor="ops@example.com"))
    do = build(gate, Stage.SUGGEST, calls)

    result = do("widget-1")

    assert calls == []
    assert result.outcome is Outcome.DENIED
    assert sink.events[0].decision is Decision.DENIED
    assert sink.events[0].duration_ms is None


def test_suggest_with_no_handler_is_refused_not_permitted(sink, calls) -> None:
    """The dangerous default would be to let it through. It does not."""
    gate = StageGate(audit=sink, approval=None)
    do = build(gate, Stage.SUGGEST, calls)

    result = do("widget-1")

    assert calls == []
    assert result.outcome is Outcome.ERROR
    assert sink.events[0].decision is Decision.ERROR
    assert "no approval handler is configured" in (sink.events[0].decision_note or "")


def test_an_approval_handler_that_raises_denies(sink, calls) -> None:
    class BrokenHandler:
        def request_approval(self, request):
            raise ConnectionError("approval service unreachable")

    gate = StageGate(audit=sink, approval=BrokenHandler())
    do = build(gate, Stage.SUGGEST, calls)

    result = do("widget-1")

    assert calls == [], "an approval system that is down cannot approve"
    assert result.outcome is Outcome.ERROR
    assert "ConnectionError" in (sink.events[0].decision_note or "")


def test_an_approval_handler_returning_junk_denies(sink, calls) -> None:
    class ConfusedHandler:
        def request_approval(self, request):
            return True  # not an ApprovalResponse

    gate = StageGate(audit=sink, approval=ConfusedHandler())
    do = build(gate, Stage.SUGGEST, calls)

    result = do("widget-1")
    assert calls == []
    assert result.outcome is Outcome.ERROR
    assert "expected ApprovalResponse" in (sink.events[0].decision_note or "")


def test_per_capability_handler_overrides_the_gate_default(sink, calls) -> None:
    gate = StageGate(audit=sink, approval=StaticApprovalHandler(True, actor="default"))
    specific = StaticApprovalHandler(False, actor="strict-reviewer")
    do = build(gate, Stage.SUGGEST, calls, approval=specific)

    result = do("widget-1")
    assert result.outcome is Outcome.DENIED
    assert sink.events[0].decision_actor == "strict-reviewer"


# ------------------------------------------------------------------- policy


def test_policy_overrides_the_stage_declared_in_code(sink, calls) -> None:
    gate = StageGate(audit=sink, policy=StagePolicy(overrides={"demo.act": Stage.OBSERVE}))
    do = build(gate, Stage.ACT, calls)

    result = do("widget-1")

    assert calls == []
    assert result.event.declared_stage is Stage.ACT
    assert result.event.stage is Stage.OBSERVE
    assert "stage from exact" in (result.event.decision_note or "")


def test_a_ceiling_lowers_every_capability(sink, calls) -> None:
    gate = StageGate(audit=sink, policy=StagePolicy(max_stage=Stage.OBSERVE))
    do = build(gate, Stage.ACT, calls)
    assert do("widget-1").outcome is Outcome.RECORDED
    assert calls == []


def test_a_ceiling_never_promotes(sink, calls) -> None:
    """max_stage is a cap, not an assignment."""
    gate = StageGate(audit=sink, policy=StagePolicy(max_stage=Stage.ACT))
    do = build(gate, Stage.OBSERVE, calls)
    assert do("widget-1").outcome is Outcome.RECORDED
    assert calls == []


def test_risk_ceiling_caps_by_tier(sink, calls) -> None:
    gate = StageGate(
        audit=sink,
        approval=StaticApprovalHandler(True),
        policy=StagePolicy(max_stage_by_risk={RiskTier.CRITICAL: Stage.SUGGEST}),
    )

    @gate.capability("demo.nuke", stage=Stage.ACT, risk=RiskTier.CRITICAL)
    def nuke() -> str:
        calls.append("nuke")
        return "boom"

    @gate.capability("demo.read", stage=Stage.ACT, risk=RiskTier.LOW)
    def read() -> str:
        return "ok"

    assert nuke().event.stage is Stage.SUGGEST
    assert read().event.stage is Stage.ACT


def test_default_stage_for_an_undeclared_capability_is_observe(sink, calls) -> None:
    """A capability nobody thought about does not act."""
    gate = StageGate(audit=sink)

    @gate.capability("demo.unthought")
    def do() -> None:
        calls.append("ran")

    assert do().outcome is Outcome.RECORDED
    assert calls == []


# ----------------------------------------------------------------- registry


def test_duplicate_registration_is_refused(gate: StageGate) -> None:
    @gate.capability("demo.dup", stage=Stage.ACT)
    def one() -> None: ...

    with pytest.raises(ConfigurationError, match="already registered"):

        @gate.capability("demo.dup", stage=Stage.ACT)
        def two() -> None: ...


def test_duplicate_registration_is_allowed_when_explicit(gate: StageGate) -> None:
    @gate.capability("demo.dup", stage=Stage.ACT)
    def one() -> str:
        return "one"

    @gate.capability("demo.dup", stage=Stage.ACT, replace=True)
    def two() -> str:
        return "two"

    assert gate.invoke("demo.dup").unwrap() == "two"


def test_async_capabilities_are_rejected_loudly(gate: StageGate) -> None:
    with pytest.raises(ConfigurationError, match="coroutine function"):

        @gate.capability("demo.async", stage=Stage.ACT)
        async def do() -> None: ...


def test_bad_stage_string_is_a_configuration_error(gate: StageGate) -> None:
    with pytest.raises(ConfigurationError, match="unknown stage"):

        @gate.capability("demo.bad", stage="sometimes")
        def do() -> None: ...


def test_invoke_dispatches_by_name_like_a_tool_call(gate: StageGate, calls) -> None:
    build(gate, Stage.ACT, calls)
    result = gate.invoke("demo.act", target="widget-9")
    assert result.unwrap() == "did widget-9"
    assert calls == ["widget-9"]


def test_invoke_on_an_unknown_name_raises_with_the_options(gate: StageGate, calls) -> None:
    build(gate, Stage.ACT, calls)
    with pytest.raises(KeyError, match=re.escape("demo.act")):
        gate.invoke("demo.nonexistent")


def test_manifest_shows_declared_versus_effective_stage(sink) -> None:
    gate = StageGate(audit=sink, policy=StagePolicy(patterns={"demo.*": Stage.OBSERVE}))

    @gate.capability("demo.thing", stage=Stage.ACT, risk=RiskTier.HIGH)
    def thing() -> None:
        """One-line summary."""

    row = gate.manifest()[0]
    assert row["declared_stage"] == "act"
    assert row["effective_stage"] == "observe"
    assert row["stage_source"] == "pattern:demo.*"
    assert row["summary"] == "One-line summary."
    assert row["has_approval_handler"] is False


def test_the_wrapper_keeps_the_original_reachable_for_testing(gate: StageGate, calls) -> None:
    do = build(gate, Stage.OBSERVE, calls)
    assert do.__wrapped__("widget-1") == "did widget-1"
    assert calls == ["widget-1"]
    assert do.capability.risk is RiskTier.MODERATE


def test_binding_failure_still_produces_a_record(gate: StageGate) -> None:
    """A TypeError from a bad call should not lose the audit record."""

    @gate.capability("demo.strict", stage=Stage.ACT)
    def strict(required: str) -> str:
        return required

    with pytest.raises(TypeError):
        strict()  # type: ignore[call-arg]

    assert len(gate.audit.events) == 1
    assert gate.audit.events[0].outcome is Outcome.FAILED


def test_result_exposes_the_run_it_belonged_to(gate: StageGate, calls) -> None:
    do = build(gate, Stage.ACT, calls)
    with agent_run("run-fixed") as run:
        result = do("widget-1")
    assert result.correlation_id == "run-fixed" == run.correlation_id
    assert result.capability == "demo.act"
    assert result.stage is Stage.ACT


# ------------------------------------------------- suppression has a limit


def test_suppressing_errors_does_not_suppress_keyboard_interrupt(sink, calls) -> None:
    """propagate_errors=False hides failures, not the operator pressing Ctrl-C."""
    gate = StageGate(audit=sink, propagate_errors=False)

    @gate.capability("demo.interruptible", stage=Stage.ACT)
    def slow() -> None:
        calls.append("started")
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        slow()

    assert calls == ["started"]
    assert len(sink.events) == 1, "the record is still written before it escapes"
    assert sink.events[0].outcome is Outcome.FAILED
    assert sink.events[0].error == {"type": "KeyboardInterrupt", "message": ""}


def test_suppressing_errors_does_not_suppress_system_exit(sink) -> None:
    gate = StageGate(audit=sink, propagate_errors=False)

    @gate.capability("demo.exiting", stage=Stage.ACT, propagate_errors=False)
    def bail() -> None:
        raise SystemExit(3)

    with pytest.raises(SystemExit):
        bail()
    assert sink.events[0].outcome is Outcome.FAILED


def test_ordinary_exceptions_are_still_suppressed(sink) -> None:
    """The narrowing above must not quietly re-enable propagation for everything."""
    gate = StageGate(audit=sink, propagate_errors=False)

    @gate.capability("demo.ordinary", stage=Stage.ACT)
    def boom() -> None:
        raise ValueError("bad input")

    result = boom()
    assert result.outcome is Outcome.FAILED
    assert isinstance(result.error, ValueError)


# ------------------------------------------------------------- manifest edges


@pytest.mark.parametrize(
    "docstring, expected",
    [
        (None, ""),
        ("", ""),
        ("   ", ""),
        ("\n\n", ""),
        ("One-line summary.", "One-line summary."),
        (
            "\n  Leading blank line, then this.\n\n  More detail.\n",
            "Leading blank line, then this.",
        ),
    ],
)
def test_manifest_summary_survives_any_docstring(sink, docstring, expected) -> None:
    """A capability with an odd docstring must not break startup introspection."""
    gate = StageGate(audit=sink)

    def thing() -> None:
        pass

    thing.__doc__ = docstring
    gate.register(thing, name="demo.doc", stage=Stage.OBSERVE)
    assert gate.manifest()[0]["summary"] == expected
