"""Approval handlers: prompting, timeouts, and every way a decision can fail."""

from __future__ import annotations

import dataclasses
import io
import threading
import time

import pytest

from conftest import make_request
from stagegate import (
    ApprovalError,
    ApprovalRequest,
    ApprovalResponse,
    CLIApprovalHandler,
    Decision,
    Outcome,
    QueueApprovalHandler,
    RiskTier,
    Stage,
    StageGate,
)

# ------------------------------------------------------------------- CLI


@pytest.mark.parametrize("answer", ["y\n", "yes\n", "YES\n", "approve\n", " ok \n", "go\n"])
def test_cli_accepts_the_usual_affirmatives(answer: str) -> None:
    handler = CLIApprovalHandler(stream_in=io.StringIO(answer), stream_out=io.StringIO())
    assert handler.request_approval(make_request()).approved is True


@pytest.mark.parametrize("answer", ["n\n", "no\n", "\n", "nope\n", "later\n", "why\n"])
def test_cli_treats_anything_else_as_a_refusal(answer: str) -> None:
    handler = CLIApprovalHandler(stream_in=io.StringIO(answer), stream_out=io.StringIO())
    response = handler.request_approval(make_request())
    assert response.approved is False
    assert "declined at prompt" in (response.note or "")


def test_cli_eof_is_a_timeout_not_an_approval() -> None:
    """Nobody is at the terminal. Silence is not consent."""
    handler = CLIApprovalHandler(stream_in=io.StringIO(""), stream_out=io.StringIO())
    with pytest.raises(TimeoutError):
        handler.request_approval(make_request())


def test_cli_expired_request_does_not_even_read() -> None:
    handler = CLIApprovalHandler(stream_in=io.StringIO("y\n"), stream_out=io.StringIO())
    stale = make_request(timeout=0.0)
    with pytest.raises(TimeoutError):
        handler.request_approval(stale)


def test_cli_late_answer_is_discarded(monkeypatch: pytest.MonkeyPatch) -> None:
    """A StringIO has no descriptor to poll, so the read cannot be interrupted.

    The handler still refuses to honour an answer that arrived after the deadline:
    the timeout degrades in responsiveness, never in its safety property.
    """
    from stagegate import approval as approval_module

    handler = CLIApprovalHandler(stream_in=io.StringIO("y\n"), stream_out=io.StringIO())
    request = ApprovalRequest(
        request_id="apr-slow",
        capability="demo.cap",
        effect="do the thing",
        arguments={},
        risk_tier=RiskTier.MODERATE,
        stage=Stage.SUGGEST,
        correlation_id="run-test",
        event_id="evt-test",
        requested_at=1000.0,
        timeout_s=10.0,
    )
    # Ticks: render, remaining(), deadline, then a post-read clock far past it.
    ticks = iter([1000.0, 1000.0, 1000.0, 9999.0])
    monkeypatch.setattr(approval_module.time, "monotonic", lambda: next(ticks, 9999.0))

    with pytest.raises(TimeoutError):
        handler.request_approval(request)


def test_cli_prompt_shows_the_effect_and_the_redacted_arguments() -> None:
    out = io.StringIO()
    handler = CLIApprovalHandler(stream_in=io.StringIO("y\n"), stream_out=out)
    handler.request_approval(make_request())
    printed = out.getvalue()
    assert "APPROVAL REQUIRED" in printed
    assert "do the thing" in printed
    assert "target = 'widget-1'" in printed
    assert "risk=moderate" in printed


def test_critical_capabilities_require_typing_the_name() -> None:
    """Muscle memory approves anything; typing the capability name does not happen by accident."""
    handler = CLIApprovalHandler(stream_in=io.StringIO("y\n"), stream_out=io.StringIO())
    assert handler.request_approval(make_request(risk=RiskTier.CRITICAL)).approved is False

    handler = CLIApprovalHandler(
        stream_in=io.StringIO("billing.refund\n"), stream_out=io.StringIO()
    )
    response = handler.request_approval(
        make_request(capability="billing.refund", risk=RiskTier.CRITICAL)
    )
    assert response.approved is True


def test_typed_confirmation_threshold_is_configurable() -> None:
    handler = CLIApprovalHandler(
        stream_in=io.StringIO("y\n"),
        stream_out=io.StringIO(),
        confirm_at_or_above=RiskTier.MODERATE,
    )
    assert handler.request_approval(make_request(risk=RiskTier.HIGH)).approved is False

    handler = CLIApprovalHandler(
        stream_in=io.StringIO("y\n"), stream_out=io.StringIO(), confirm_at_or_above=None
    )
    assert handler.request_approval(make_request(risk=RiskTier.CRITICAL)).approved is True


def test_cli_records_who_approved() -> None:
    handler = CLIApprovalHandler(
        stream_in=io.StringIO("y\n"), stream_out=io.StringIO(), actor="ada@example.com"
    )
    assert handler.request_approval(make_request()).actor == "ada@example.com"


def test_cli_actor_falls_back_to_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STAGEGATE_ACTOR", "oncall@example.com")
    handler = CLIApprovalHandler(stream_in=io.StringIO("y\n"), stream_out=io.StringIO())
    assert handler.request_approval(make_request()).actor == "oncall@example.com"


def test_cli_reports_an_unreadable_stream_as_an_approval_error() -> None:
    closed = io.StringIO()
    closed.close()
    handler = CLIApprovalHandler(stream_in=closed, stream_out=io.StringIO())
    with pytest.raises(ApprovalError):
        handler.request_approval(make_request())


def test_cli_survives_an_unwritable_prompt_stream() -> None:
    closed = io.StringIO()
    closed.close()
    handler = CLIApprovalHandler(stream_in=io.StringIO("y\n"), stream_out=closed)
    with pytest.raises(ApprovalError, match="cannot write approval prompt"):
        handler.request_approval(make_request())


# ----------------------------------------------------------------- queue


def test_queue_handler_blocks_until_another_thread_resolves() -> None:
    handler = QueueApprovalHandler()
    result: list[ApprovalResponse] = []

    def agent() -> None:
        result.append(handler.request_approval(make_request(timeout=5.0)))

    thread = threading.Thread(target=agent)
    thread.start()

    request = handler.next_request(timeout=2.0)
    assert request is not None
    delivered = handler.resolve(request.request_id, True, actor="ops@example.com", note="checked")
    assert delivered is True

    thread.join(timeout=5.0)
    assert result[0].approved is True
    assert result[0].actor == "ops@example.com"
    assert result[0].note == "checked"


def test_queue_handler_times_out_and_fails_closed() -> None:
    handler = QueueApprovalHandler()
    started = time.monotonic()
    with pytest.raises(TimeoutError):
        handler.request_approval(make_request(timeout=0.1))
    assert time.monotonic() - started < 2.0
    assert handler.pending() == [], "a timed-out request is not left dangling"


def test_a_decision_arriving_after_the_timeout_is_refused() -> None:
    """The agent has moved on and the log says the approval never arrived."""
    handler = QueueApprovalHandler()
    with pytest.raises(TimeoutError):
        handler.request_approval(make_request(timeout=0.05))
    assert handler.resolve("apr-test", True, actor="late@example.com") is False


def test_resolving_an_unknown_request_returns_false() -> None:
    assert QueueApprovalHandler().resolve("nope", True) is False


def test_a_request_can_only_be_decided_once() -> None:
    handler = QueueApprovalHandler()
    responses: list[ApprovalResponse] = []
    thread = threading.Thread(
        target=lambda: responses.append(handler.request_approval(make_request(timeout=5.0)))
    )
    thread.start()
    request = handler.next_request(timeout=2.0)
    assert request is not None
    assert handler.resolve(request.request_id, True, actor="first") is True
    thread.join(timeout=5.0)
    assert handler.resolve(request.request_id, False, actor="second") is False
    assert responses[0].actor == "first"


def _call_and_capture(handler, request) -> BaseException | None:
    """Run ``request_approval`` on a worker thread and hand the exception back.

    A bare ``pytest.raises`` inside a thread target loses its AssertionError with
    the thread, so the test could not fail when the call unexpectedly *succeeded*.
    The main thread asserts on the returned exception instead.
    """
    try:
        handler.request_approval(request)
    except BaseException as exc:  # handed back to the thread that can assert on it
        return exc
    return None


def test_pending_exposes_what_is_waiting() -> None:
    handler = QueueApprovalHandler()
    outcome: list[BaseException | None] = []
    thread = threading.Thread(
        target=lambda: outcome.append(_call_and_capture(handler, make_request(timeout=0.5)))
    )
    thread.start()
    deadline = time.monotonic() + 2.0
    while not handler.pending() and time.monotonic() < deadline:
        time.sleep(0.005)
    pending = handler.pending()
    assert len(pending) == 1
    assert pending[0].request.capability == "demo.cap"
    assert pending[0].request.summary().startswith("[moderate] demo.cap")
    thread.join(timeout=5.0)
    assert not thread.is_alive()
    assert isinstance(outcome[0], TimeoutError), "an unanswered request must time out"


def test_deny_all_is_what_an_operator_reaches_for() -> None:
    handler = QueueApprovalHandler()
    outcomes: list[bool] = []

    def agent(index: int) -> None:
        request = dataclasses.replace(make_request(timeout=5.0), request_id=f"apr-{index}")
        try:
            outcomes.append(handler.request_approval(request).approved)
        except TimeoutError:  # pragma: no cover - would mean deny_all was too slow
            outcomes.append(True)

    threads = [threading.Thread(target=agent, args=(i,)) for i in range(3)]
    for thread in threads:
        thread.start()
    deadline = time.monotonic() + 3.0
    while len(handler.pending()) < 3 and time.monotonic() < deadline:
        time.sleep(0.005)

    assert handler.deny_all(actor="incident-commander") == 3
    for thread in threads:
        thread.join(timeout=5.0)
    assert outcomes == [False, False, False]


def test_a_notification_hook_that_raises_does_not_become_an_approval() -> None:
    def broken(request) -> None:
        raise ConnectionError("pager down")

    handler = QueueApprovalHandler(on_submit=broken)
    with pytest.raises(TimeoutError):
        handler.request_approval(make_request(timeout=0.05))
    assert len(handler.notify_errors) == 1


def test_on_submit_receives_the_request() -> None:
    seen: list[str] = []
    handler = QueueApprovalHandler(on_submit=lambda r: seen.append(r.summary()))
    with pytest.raises(TimeoutError):
        handler.request_approval(make_request(timeout=0.05))
    assert seen == ["[moderate] demo.cap: do the thing"]


def test_next_request_returns_none_when_nothing_arrives() -> None:
    assert QueueApprovalHandler().next_request(timeout=0.05) is None


# ------------------------------------------------------ integrated timeout


def test_a_gate_records_an_approval_timeout_as_a_refusal(sink, calls) -> None:
    gate = StageGate(audit=sink, approval=QueueApprovalHandler(), approval_timeout=0.1)

    @gate.capability("demo.slow", stage=Stage.SUGGEST, describe="act on {target}")
    def slow(target: str) -> str:
        calls.append(target)
        return "done"

    result = slow("widget-1")

    assert calls == []
    assert result.outcome is Outcome.TIMED_OUT
    assert sink.events[0].decision is Decision.TIMED_OUT
    assert sink.events[0].approval_latency_ms is not None
    assert sink.events[0].effect == "act on widget-1"


def test_a_per_capability_timeout_overrides_the_gate_default(sink, calls) -> None:
    gate = StageGate(audit=sink, approval=QueueApprovalHandler(), approval_timeout=30.0)

    @gate.capability("demo.impatient", stage=Stage.SUGGEST, approval_timeout=0.05)
    def impatient() -> None:
        calls.append("ran")

    started = time.monotonic()
    result = impatient()
    assert time.monotonic() - started < 5.0
    assert result.outcome is Outcome.TIMED_OUT
    assert calls == []


def test_the_request_a_handler_sees_carries_the_run_identity(sink) -> None:
    from stagegate import agent_run

    seen: list = []

    class Recorder:
        def request_approval(self, request):
            seen.append(request)
            return ApprovalResponse(True, actor="ops")

    gate = StageGate(audit=sink, approval=Recorder())

    @gate.capability("demo.cap", stage=Stage.SUGGEST, risk=RiskTier.HIGH)
    def cap(target: str) -> None: ...

    with agent_run("run-abc", actor="bot@example.com", labels={"tenant": "acme"}):
        cap("widget-1")

    request = seen[0]
    assert request.correlation_id == "run-abc"
    assert request.actor == "bot@example.com"
    assert request.labels == {"tenant": "acme"}
    assert request.risk_tier is RiskTier.HIGH
    assert request.stage is Stage.SUGGEST
    assert request.remaining() >= 0


def test_a_duplicate_pending_request_id_is_refused() -> None:
    """Two in-flight requests sharing an id would let one decision resolve the wrong call."""
    handler = QueueApprovalHandler()
    outcome: list[BaseException | None] = []
    thread = threading.Thread(
        target=lambda: outcome.append(_call_and_capture(handler, make_request(timeout=1.0)))
    )
    thread.start()
    deadline = time.monotonic() + 2.0
    while not handler.pending() and time.monotonic() < deadline:
        time.sleep(0.005)

    with pytest.raises(ApprovalError, match="already pending"):
        handler.request_approval(make_request(timeout=1.0))
    thread.join(timeout=5.0)
    assert not thread.is_alive()
    assert isinstance(outcome[0], TimeoutError), (
        "the first waiter must keep its own timeout, not be resolved by the duplicate"
    )


def test_a_gate_records_a_duplicate_request_id_error_as_a_refusal(sink, calls) -> None:
    """The gate turns any ApprovalError into a recorded refusal, never an execution."""

    class AlwaysBroken:
        def request_approval(self, request):
            raise ApprovalError("approval store unavailable")

    gate = StageGate(audit=sink, approval=AlwaysBroken())

    @gate.capability("demo.cap", stage=Stage.SUGGEST)
    def cap() -> None:
        calls.append("ran")

    result = cap()
    assert calls == []
    assert result.outcome is Outcome.ERROR
    assert "approval store unavailable" in (sink.events[0].decision_note or "")


# ------------------------------------------------- bounded in a long-lived process


def test_arrival_hints_do_not_grow_without_bound_when_nobody_polls() -> None:
    """pending() is the source of truth; the hint queue must not leak.

    A production handler outlives every request it serves. If a deployment reads
    ``pending()`` on a timer instead of calling ``next_request()``, nothing drains
    the arrival queue, and an unbounded one is a leak that only shows up in week
    three of a rollout.
    """
    handler = QueueApprovalHandler(history=4)

    for index in range(200):
        handler._record_arrival(f"apr-{index}")

    assert handler._arrivals.qsize() == 4
    kept = [handler._arrivals.get_nowait() for _ in range(4)]
    assert kept == ["apr-196", "apr-197", "apr-198", "apr-199"], "the newest hints survive"


def test_notification_failures_are_capped_at_the_history_size() -> None:
    def always_broken(request: ApprovalRequest) -> None:
        raise ConnectionError("pager unreachable")

    handler = QueueApprovalHandler(default_timeout=0.01, history=3, on_submit=always_broken)
    for index in range(10):
        with pytest.raises(TimeoutError):
            handler.request_approval(
                dataclasses.replace(make_request(timeout=0.01), request_id=f"apr-{index}")
            )

    assert len(handler.notify_errors) == 3, "oldest failures are dropped, not accumulated"
    assert [request_id for request_id, _ in handler.notify_errors] == [
        "apr-7",
        "apr-8",
        "apr-9",
    ]
    assert all(isinstance(exc, ConnectionError) for _, exc in handler.notify_errors)
