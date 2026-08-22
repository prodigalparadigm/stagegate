"""Correlation ids: one run, one id, and a parent pointer across nested calls."""

from __future__ import annotations

import threading

from stagegate import Stage, StageGate, agent_run, current_correlation_id, current_run


def test_calls_in_one_run_share_a_correlation_id(gate: StageGate, sink) -> None:
    @gate.capability("demo.one", stage=Stage.ACT)
    def one() -> None: ...

    @gate.capability("demo.two", stage=Stage.ACT)
    def two() -> None: ...

    with agent_run() as run:
        one()
        two()

    assert {e.correlation_id for e in sink.events} == {run.correlation_id}


def test_separate_runs_get_separate_ids(gate: StageGate, sink) -> None:
    @gate.capability("demo.one", stage=Stage.ACT)
    def one() -> None: ...

    with agent_run():
        one()
    with agent_run():
        one()

    assert len({e.correlation_id for e in sink.events}) == 2


def test_a_call_outside_any_run_still_gets_an_id(gate: StageGate, sink) -> None:
    @gate.capability("demo.one", stage=Stage.ACT)
    def one() -> None: ...

    one()
    assert sink.events[0].correlation_id.startswith("run-")


def test_an_upstream_id_can_be_adopted(gate: StageGate, sink) -> None:
    """Stitches the audit trail to whatever tracing the rest of the system uses."""
    @gate.capability("demo.one", stage=Stage.ACT)
    def one() -> None: ...

    with agent_run("req-from-the-gateway"):
        one()

    assert sink.events[0].correlation_id == "req-from-the-gateway"


def test_nested_calls_point_at_their_caller(gate: StageGate, sink) -> None:
    @gate.capability("demo.inner", stage=Stage.ACT)
    def inner() -> None: ...

    @gate.capability("demo.outer", stage=Stage.ACT)
    def outer() -> None:
        inner()

    with agent_run("run-1"):
        outer()

    events = {e.capability: e for e in sink.events}
    assert events["demo.inner"].parent_event_id == events["demo.outer"].event_id
    assert events["demo.outer"].parent_event_id is None
    assert events["demo.inner"].depth == 1
    assert events["demo.outer"].depth == 0
    assert events["demo.inner"].correlation_id == "run-1"


def test_three_levels_of_nesting_form_a_chain(gate: StageGate, sink) -> None:
    @gate.capability("demo.c", stage=Stage.ACT)
    def c() -> None: ...

    @gate.capability("demo.b", stage=Stage.ACT)
    def b() -> None:
        c()

    @gate.capability("demo.a", stage=Stage.ACT)
    def a() -> None:
        b()

    with agent_run("run-1"):
        a()

    events = {e.capability: e for e in sink.events}
    assert events["demo.b"].parent_event_id == events["demo.a"].event_id
    assert events["demo.c"].parent_event_id == events["demo.b"].event_id
    assert [events[n].depth for n in ("demo.a", "demo.b", "demo.c")] == [0, 1, 2]


def test_the_parent_scope_is_restored_after_a_nested_call(gate: StageGate, sink) -> None:
    """Two siblings must both point at the parent, not at each other."""
    @gate.capability("demo.leaf", stage=Stage.ACT)
    def leaf() -> None: ...

    @gate.capability("demo.root", stage=Stage.ACT)
    def root() -> None:
        leaf()
        leaf.__wrapped__()  # not gated
        leaf()

    with agent_run("run-1"):
        root()

    root_event = next(e for e in sink.events if e.capability == "demo.root")
    leaves = [e for e in sink.events if e.capability == "demo.leaf"]
    assert len(leaves) == 2
    assert {e.parent_event_id for e in leaves} == {root_event.event_id}


def test_a_nested_call_after_a_failure_is_still_scoped(gate: StageGate, sink) -> None:
    @gate.capability("demo.leaf", stage=Stage.ACT, propagate_errors=False)
    def leaf(boom: bool) -> None:
        if boom:
            raise RuntimeError("nope")

    @gate.capability("demo.root", stage=Stage.ACT)
    def root() -> None:
        leaf(True)
        leaf(False)

    with agent_run("run-1"):
        root()

    root_event = next(e for e in sink.events if e.capability == "demo.root")
    leaves = [e for e in sink.events if e.capability == "demo.leaf"]
    assert {e.parent_event_id for e in leaves} == {root_event.event_id}


def test_reentering_a_run_reuses_it_rather_than_splitting_the_trail() -> None:
    with agent_run("run-1") as outer, agent_run() as inner:
        assert inner is outer
        assert current_correlation_id() == "run-1"


def test_a_sub_run_can_be_requested_explicitly() -> None:
    with agent_run("run-1"), agent_run(nest=True) as inner:
        assert inner.correlation_id != "run-1"
        assert inner.depth == 1
    assert current_run() is None


def test_the_run_scope_is_cleaned_up_after_an_exception() -> None:
    try:
        with agent_run("run-1"):
            raise ValueError("boom")
    except ValueError:
        pass
    assert current_run() is None


def test_labels_and_actor_ride_along_on_every_event(gate: StageGate, sink) -> None:
    @gate.capability("demo.one", stage=Stage.ACT)
    def one() -> None: ...

    with agent_run(actor="bot@example.com", labels={"tenant": "acme"}):
        one()

    assert sink.events[0].actor == "bot@example.com"
    assert sink.events[0].labels == {"tenant": "acme"}


def test_gate_labels_merge_with_run_labels(sink) -> None:
    gate = StageGate(audit=sink, labels={"deployment": "prod", "tenant": "default"})

    @gate.capability("demo.one", stage=Stage.ACT)
    def one() -> None: ...

    with agent_run(labels={"tenant": "acme"}):
        one()

    assert sink.events[0].labels == {"deployment": "prod", "tenant": "acme"}


def test_concurrent_runs_in_separate_threads_do_not_bleed(gate: StageGate, sink) -> None:
    @gate.capability("demo.one", stage=Stage.ACT)
    def one() -> None: ...

    def worker(run_id: str) -> None:
        with agent_run(run_id):
            for _ in range(5):
                one()

    threads = [threading.Thread(target=worker, args=(f"run-{i}",)) for i in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    from collections import Counter

    counts = Counter(e.correlation_id for e in sink.events)
    assert counts == {f"run-{i}": 5 for i in range(4)}


def test_the_gate_actor_is_used_when_the_run_supplies_none(sink) -> None:
    gate = StageGate(audit=sink, actor="fallback@example.com")

    @gate.capability("demo.one", stage=Stage.ACT)
    def one() -> None: ...

    with agent_run():
        one()

    assert sink.events[0].actor == "fallback@example.com"
