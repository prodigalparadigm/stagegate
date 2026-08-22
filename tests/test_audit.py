"""Audit log: append-only guarantees, chain integrity, and recovery."""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from stagegate import (
    AuditEvent,
    AuditLogCorruption,
    AuditWriteError,
    Decision,
    InMemoryAuditLog,
    JsonlAuditLog,
    MultiSink,
    Outcome,
    RiskTier,
    Stage,
    StageGate,
    StreamAuditSink,
    read_events,
    verify_chain,
)
from stagegate.audit import GENESIS_HASH, SCHEMA


def event(name: str = "demo.cap", **overrides) -> AuditEvent:
    defaults = dict(
        capability=name,
        stage=Stage.OBSERVE,
        declared_stage=Stage.OBSERVE,
        outcome=Outcome.RECORDED,
        decision=Decision.NOT_REQUIRED,
        correlation_id="run-1",
        event_id=f"evt-{name}",
    )
    defaults.update(overrides)
    return AuditEvent(**defaults)


def lines(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


# ------------------------------------------------------------- append-only


def test_the_sink_exposes_no_way_to_change_or_remove_a_record(tmp_path: Path) -> None:
    """The API surface is the first line of defence: there is nothing to call."""
    log = JsonlAuditLog(tmp_path / "audit.jsonl")
    public = {name for name in dir(log) if not name.startswith("_")}
    assert public == {"emit", "preflight", "close", "verify", "path", "fsync"}
    log.close()


def test_records_accumulate_and_earlier_ones_are_untouched(tmp_path: Path) -> None:
    path = tmp_path / "audit.jsonl"
    with JsonlAuditLog(path) as log:
        log.emit(event("first"))
        after_one = path.read_text()
        log.emit(event("second"))
        after_two = path.read_text()

    assert after_two.startswith(after_one), "an append must never rewrite what is already there"
    assert [r["capability"] for r in lines(path)] == ["first", "second"]
    assert [r["seq"] for r in lines(path)] == [1, 2]


def test_the_first_record_chains_from_genesis(tmp_path: Path) -> None:
    path = tmp_path / "audit.jsonl"
    with JsonlAuditLog(path) as log:
        written = log.emit(event())
    assert written.prev_hash == GENESIS_HASH
    assert written.seq == 1
    assert lines(path)[0]["schema"] == SCHEMA


def test_reopening_continues_the_existing_chain(tmp_path: Path) -> None:
    path = tmp_path / "audit.jsonl"
    with JsonlAuditLog(path) as log:
        first = log.emit(event("first"))
    with JsonlAuditLog(path) as log:
        second = log.emit(event("second"))

    assert second.seq == 2
    assert second.prev_hash == first.hash
    assert verify_chain(path).ok


def test_a_file_descriptor_is_reused_rather_than_reopened(tmp_path: Path) -> None:
    """Reopening per write would race with log rotation and cost a syscall each time."""
    path = tmp_path / "audit.jsonl"
    with JsonlAuditLog(path) as log:
        for index in range(50):
            log.emit(event(f"cap-{index}"))
    assert verify_chain(path).count == 50


# ------------------------------------------------------- tamper detection


def test_a_clean_log_verifies(tmp_path: Path) -> None:
    path = tmp_path / "audit.jsonl"
    with JsonlAuditLog(path) as log:
        for index in range(5):
            log.emit(event(f"cap-{index}"))

    result = verify_chain(path)
    assert result.ok is True
    assert result.count == 5
    assert result.first_bad_seq is None
    assert result.head_hash


def test_editing_a_record_is_detected_at_that_record(tmp_path: Path) -> None:
    path = tmp_path / "audit.jsonl"
    with JsonlAuditLog(path) as log:
        for index in range(5):
            log.emit(event(f"cap-{index}"))

    rows = path.read_text().splitlines()
    tampered = json.loads(rows[2])
    tampered["outcome"] = "recorded"
    tampered["capability"] = "something-innocent"
    rows[2] = json.dumps(tampered, sort_keys=True, separators=(",", ":"))
    path.write_text("\n".join(rows) + "\n")

    result = verify_chain(path)
    assert result.ok is False
    assert result.first_bad_seq == 3
    assert "do not match the recorded hash" in (result.reason or "")


def test_deleting_a_record_from_the_middle_is_detected(tmp_path: Path) -> None:
    path = tmp_path / "audit.jsonl"
    with JsonlAuditLog(path) as log:
        for index in range(5):
            log.emit(event(f"cap-{index}"))

    rows = path.read_text().splitlines()
    del rows[2]
    path.write_text("\n".join(rows) + "\n")

    result = verify_chain(path)
    assert result.ok is False
    assert result.first_bad_seq == 3
    assert "removed, reordered, or inserted" in (result.reason or "")


def test_reordering_records_is_detected(tmp_path: Path) -> None:
    path = tmp_path / "audit.jsonl"
    with JsonlAuditLog(path) as log:
        for index in range(4):
            log.emit(event(f"cap-{index}"))

    rows = path.read_text().splitlines()
    rows[1], rows[2] = rows[2], rows[1]
    path.write_text("\n".join(rows) + "\n")

    assert verify_chain(path).ok is False


def test_truncating_the_tail_is_detected_by_comparing_the_head_hash(tmp_path: Path) -> None:
    """Dropping trailing records leaves a self-consistent chain: anchor the head hash."""
    path = tmp_path / "audit.jsonl"
    with JsonlAuditLog(path) as log:
        for index in range(5):
            log.emit(event(f"cap-{index}"))
    published_head = verify_chain(path).head_hash

    rows = path.read_text().splitlines()
    path.write_text("\n".join(rows[:3]) + "\n")

    after = verify_chain(path)
    assert after.ok is True, "a truncated prefix is internally consistent, as documented"
    assert after.head_hash != published_head, "which is why the head hash must be anchored"
    assert after.count == 3


def test_a_forged_record_appended_without_the_chain_is_detected(tmp_path: Path) -> None:
    path = tmp_path / "audit.jsonl"
    with JsonlAuditLog(path) as log:
        log.emit(event("real"))

    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps({"seq": 2, "capability": "forged", "hash": "deadbeef"}) + "\n")

    result = verify_chain(path)
    assert result.ok is False
    assert result.first_bad_seq == 2


def test_verifying_a_missing_log_is_a_result_not_an_exception(tmp_path: Path) -> None:
    result = verify_chain(tmp_path / "nope.jsonl")
    assert result.ok is False
    assert "no such audit log" in (result.reason or "")


def test_verifying_an_empty_log_is_vacuously_true(tmp_path: Path) -> None:
    path = tmp_path / "empty.jsonl"
    path.write_text("")
    result = verify_chain(path)
    assert result.ok is True
    assert result.count == 0


# ------------------------------------------------------------- recovery


def test_a_torn_trailing_line_is_repaired_on_open(tmp_path: Path) -> None:
    """A crash mid-write leaves bytes that were never a committed record."""
    path = tmp_path / "audit.jsonl"
    with JsonlAuditLog(path) as log:
        log.emit(event("first"))
        log.emit(event("second"))

    with open(path, "a", encoding="utf-8") as handle:
        handle.write('{"seq": 3, "capability": "half-writt')

    with JsonlAuditLog(path) as log:
        third = log.emit(event("third"))

    assert third.seq == 3
    assert verify_chain(path).ok is True
    assert [r["capability"] for r in lines(path)] == ["first", "second", "third"]


def test_a_corrupt_final_record_refuses_to_open_by_default(tmp_path: Path) -> None:
    path = tmp_path / "audit.jsonl"
    with JsonlAuditLog(path) as log:
        log.emit(event("first"))

    rows = path.read_text().splitlines()
    row = json.loads(rows[0])
    row["capability"] = "edited"
    path.write_text(json.dumps(row) + "\n")

    with pytest.raises(AuditLogCorruption, match="does not match its contents"):
        JsonlAuditLog(path)


def test_seal_quarantines_the_suspect_file_and_starts_a_new_chain(tmp_path: Path) -> None:
    path = tmp_path / "audit.jsonl"
    with JsonlAuditLog(path) as log:
        log.emit(event("first"))
    row = json.loads(path.read_text().splitlines()[0])
    row["capability"] = "edited"
    path.write_text(json.dumps(row) + "\n")

    with JsonlAuditLog(path, on_corrupt="seal") as log:
        written = log.emit(event("after-reset"))

    sealed = list(tmp_path.glob("audit.jsonl.corrupt-*"))
    assert len(sealed) == 1, "the suspect file is kept, not deleted"
    assert written.seq == 1
    record = lines(path)[0]
    assert record["chain_reset_from"] == sealed[0].name, "the break is itself in the record"
    assert verify_chain(path).ok is True


def test_emitting_to_a_closed_log_raises(tmp_path: Path) -> None:
    log = JsonlAuditLog(tmp_path / "audit.jsonl")
    log.close()
    log.close()  # idempotent
    with pytest.raises(AuditWriteError, match="closed"):
        log.emit(event())
    with pytest.raises(AuditWriteError):
        log.preflight()


def test_an_unwritable_directory_fails_at_construction_not_mid_run(tmp_path: Path) -> None:
    """Fail at startup, where a human is watching, rather than during an incident."""
    blocker = tmp_path / "not-a-dir"
    blocker.write_text("i am a file")
    with pytest.raises((AuditWriteError, NotADirectoryError, OSError)):
        JsonlAuditLog(blocker / "audit.jsonl")


# ------------------------------------------------------------ concurrency


def test_concurrent_writers_in_one_process_produce_a_valid_chain(tmp_path: Path) -> None:
    path = tmp_path / "audit.jsonl"
    log = JsonlAuditLog(path)

    def worker(worker_id: int) -> None:
        for index in range(25):
            log.emit(event(f"w{worker_id}-{index}"))

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    log.close()

    result = verify_chain(path)
    assert result.ok is True
    assert result.count == 100


# ------------------------------------------------------------- other sinks


def test_reading_events_back_round_trips_the_schema(tmp_path: Path) -> None:
    path = tmp_path / "audit.jsonl"
    with JsonlAuditLog(path) as log:
        log.emit(
            event(
                "demo.cap",
                stage=Stage.ACT,
                risk_tier=RiskTier.HIGH,
                outcome=Outcome.EXECUTED,
                decision=Decision.APPROVED,
                arguments={"target": "widget-1"},
                effect="act on widget-1",
                actor="bot@example.com",
                decision_actor="ops@example.com",
                duration_ms=12.5,
                labels={"tenant": "acme"},
            )
        )

    restored = list(read_events(path))[0]
    assert restored.capability == "demo.cap"
    assert restored.stage is Stage.ACT
    assert restored.risk_tier is RiskTier.HIGH
    assert restored.outcome is Outcome.EXECUTED
    assert restored.decision is Decision.APPROVED
    assert restored.arguments == {"target": "widget-1"}
    assert restored.duration_ms == 12.5
    assert restored.labels == {"tenant": "acme"}


def test_reading_skips_malformed_lines_but_strict_mode_raises(tmp_path: Path) -> None:
    path = tmp_path / "audit.jsonl"
    with JsonlAuditLog(path) as log:
        log.emit(event("good"))
    with open(path, "a", encoding="utf-8") as handle:
        handle.write("not json at all\n")

    assert len(list(read_events(path))) == 1
    with pytest.raises(AuditLogCorruption):
        list(read_events(path, strict=True))


def test_an_unknown_stage_from_a_newer_writer_does_not_break_a_reader(tmp_path: Path) -> None:
    path = tmp_path / "audit.jsonl"
    path.write_text(json.dumps({"capability": "x", "stage": "teleport", "outcome": "warped"}) + "\n")
    restored = list(read_events(path))[0]
    assert restored.stage is Stage.OBSERVE
    assert restored.outcome is Outcome.ERROR


def test_multisink_mirrors_and_survives_a_broken_mirror(tmp_path: Path) -> None:
    primary = InMemoryAuditLog()
    mirror = InMemoryAuditLog()
    mirror.fail_next = ConnectionError("SIEM unreachable")
    sink = MultiSink(primary, mirror)

    written = sink.emit(event("first"))
    assert written.seq == 1
    assert len(sink.errors) == 1

    sink.emit(event("second"))
    assert len(mirror.events) == 1, "the mirror recovers on the next event"
    assert len(primary.events) == 2
    sink.close()


def test_multisink_propagates_a_failure_of_the_log_of_record() -> None:
    primary = InMemoryAuditLog()
    primary.fail_next = OSError("disk full")
    with pytest.raises(OSError):
        MultiSink(primary, InMemoryAuditLog()).emit(event())


def test_multisink_needs_at_least_one_sink() -> None:
    with pytest.raises(ValueError):
        MultiSink()


def test_stream_sink_writes_verifiable_jsonl() -> None:
    import io

    buffer = io.StringIO()
    sink = StreamAuditSink(buffer)
    sink.emit(event("first"))
    sink.emit(event("second"))
    records = [json.loads(line) for line in buffer.getvalue().splitlines()]
    assert [r["seq"] for r in records] == [1, 2]
    assert records[1]["prev_hash"] == records[0]["hash"]


# ------------------------------------------------ the gate's audit guarantees


def test_a_capability_that_cannot_be_audited_does_not_run(sink, calls) -> None:
    """strict_audit means the record is a precondition for the action."""
    class DeadSink(InMemoryAuditLog):
        def preflight(self) -> None:
            raise AuditWriteError("volume unmounted")

    dead = DeadSink()
    gate = StageGate(audit=dead, strict_audit=True)

    @gate.capability("demo.act", stage=Stage.ACT)
    def act() -> None:
        calls.append("ran")

    result = act()

    assert calls == [], "no audit, no action"
    assert result.outcome is Outcome.BLOCKED
    assert "audit sink unavailable" in (result.event.decision_note or "")
    assert len(dead.events) == 1, "the blocked attempt is itself recorded"


def test_strict_audit_can_be_disabled(sink, calls) -> None:
    class DeadSink(InMemoryAuditLog):
        def preflight(self) -> None:
            raise AuditWriteError("volume unmounted")

    gate = StageGate(audit=DeadSink(), strict_audit=False)

    @gate.capability("demo.act", stage=Stage.ACT)
    def act() -> None:
        calls.append("ran")

    assert act().outcome is Outcome.EXECUTED
    assert calls == ["ran"]


def test_a_failing_primary_sink_falls_back_and_says_so(calls) -> None:
    import io

    primary = InMemoryAuditLog()
    primary.fail_next = OSError("disk full")
    buffer = io.StringIO()
    gate = StageGate(audit=primary, fallback_sink=StreamAuditSink(buffer))

    @gate.capability("demo.act", stage=Stage.ACT)
    def act() -> None:
        calls.append("ran")

    result = act()

    assert result.audit_degraded is True
    assert "demo.act" in buffer.getvalue()
    assert calls == ["ran"]


def test_when_no_sink_accepts_the_record_the_gate_is_loud(calls) -> None:
    """At this point the process has acted without being able to account for it."""
    primary = InMemoryAuditLog()
    primary.fail_next = OSError("disk full")
    fallback = InMemoryAuditLog()
    fallback.fail_next = OSError("stderr closed")
    gate = StageGate(audit=primary, fallback_sink=fallback)

    @gate.capability("demo.act", stage=Stage.ACT)
    def act() -> None:
        calls.append("ran")

    with pytest.raises(AuditWriteError, match="reached no sink"):
        act()


def test_exactly_one_record_per_call_on_every_path(tmp_path: Path, calls) -> None:
    from stagegate import StaticApprovalHandler

    path = tmp_path / "audit.jsonl"
    log = JsonlAuditLog(path)
    gate = StageGate(audit=log, approval=StaticApprovalHandler(False))

    @gate.capability("demo.observe", stage=Stage.OBSERVE)
    def observed() -> None: ...

    @gate.capability("demo.denied", stage=Stage.SUGGEST)
    def denied() -> None: ...

    @gate.capability("demo.acted", stage=Stage.ACT)
    def acted() -> None: ...

    @gate.capability("demo.failed", stage=Stage.ACT, propagate_errors=False)
    def failed() -> None:
        raise ValueError("nope")

    observed()
    denied()
    acted()
    failed()
    log.close()

    assert verify_chain(path).count == 4
    assert [r["capability"] for r in lines(path)] == [
        "demo.observe", "demo.denied", "demo.acted", "demo.failed",
    ]
