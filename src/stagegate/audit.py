"""Append-only structured audit log with a tamper-evident hash chain.

The schema is stable and versioned (``stagegate.audit/1``). One JSON object per
line, UTF-8, newline-terminated -- readable by ``jq``, loadable by every log
pipeline, and diffable in review.

**What "append-only" actually means here.** Nothing in this module can rewrite or
delete a record: the file is opened ``O_APPEND``, and the sink exposes no update
or delete operation. But a filesystem is not a ledger, and anyone with write
access to the file can edit it behind the library's back. So each record also
carries ``prev_hash`` and ``hash``, chaining every record to its predecessor::

    hash(n) = sha256(prev_hash(n) || canonical_json(record(n) without hash))

Editing a field, reordering records, or deleting one from the middle breaks the
chain at that point, and :func:`verify_chain` says exactly where. This makes
tampering *evident*, not impossible -- the honest framing. Making it impossible
needs the digests anchored somewhere the same operator does not control, which
is a deployment decision, not a library one (see the README).
"""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import json
import os
import sys
import threading
from collections import deque
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Protocol, TextIO, runtime_checkable

from .errors import AuditLogCorruption, AuditWriteError
from .stages import Decision, Outcome, RiskTier, Stage

__all__ = [
    "SCHEMA",
    "GENESIS_HASH",
    "AuditEvent",
    "AuditSink",
    "JsonlAuditLog",
    "InMemoryAuditLog",
    "StreamAuditSink",
    "MultiSink",
    "ChainVerification",
    "verify_chain",
    "read_events",
    "canonical_json",
    "utc_now",
]

SCHEMA = "stagegate.audit/1"
"""Schema identifier written on every record. Bump only on a breaking change."""

GENESIS_HASH = "0" * 64
"""``prev_hash`` of the first record in a chain."""


def utc_now() -> str:
    """Current time as RFC 3339 UTC with a ``Z`` suffix, millisecond precision."""
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def canonical_json(obj: Any) -> str:
    """Serialise deterministically: sorted keys, no incidental whitespace.

    Determinism is what makes the hash chain reproducible on a different machine
    with a different dict ordering.
    """
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


@dataclass(frozen=True, slots=True)
class AuditEvent:
    """One capability invocation, start to finish.

    Every field is populated by the gate; construct these directly only in tests.
    ``seq``, ``prev_hash`` and ``hash`` are assigned by the sink at write time,
    because only the sink knows the chain's position.
    """

    capability: str
    stage: Stage
    """Stage actually in force, after policy ceilings and kill-switch degradation."""
    declared_stage: Stage
    """Stage the capability was registered with, before any override."""
    outcome: Outcome
    decision: Decision
    correlation_id: str
    event_id: str
    timestamp: str = field(default_factory=utc_now)
    risk_tier: RiskTier = RiskTier.MODERATE
    arguments: Mapping[str, Any] = field(default_factory=dict)
    """Already redacted. Raw values never reach this object."""
    effect: str = ""
    """Human-readable statement of what the call intended to do."""
    actor: str | None = None
    decision_actor: str | None = None
    decision_note: str | None = None
    approval_latency_ms: float | None = None
    duration_ms: float | None = None
    """Wall time of the underlying function only; ``None`` when nothing executed."""
    parent_event_id: str | None = None
    depth: int = 0
    error: Mapping[str, Any] | None = None
    kill_switch: Mapping[str, Any] | None = None
    degraded_from: Stage | None = None
    """Set when a guard forced a lower stage than policy resolved."""
    labels: Mapping[str, str] | None = None
    seq: int | None = None
    prev_hash: str | None = None
    hash: str | None = None

    def to_record(self) -> dict[str, Any]:
        """Return the JSON-ready record, minus chain fields the sink assigns."""
        record: dict[str, Any] = {
            "schema": SCHEMA,
            "timestamp": self.timestamp,
            "event_id": self.event_id,
            "correlation_id": self.correlation_id,
            "parent_event_id": self.parent_event_id,
            "depth": self.depth,
            "actor": self.actor,
            "capability": self.capability,
            "stage": self.stage.value,
            "declared_stage": self.declared_stage.value,
            "degraded_from": self.degraded_from.value if self.degraded_from else None,
            "risk_tier": self.risk_tier.value,
            "effect": self.effect,
            "arguments": dict(self.arguments),
            "decision": self.decision.value,
            "decision_actor": self.decision_actor,
            "decision_note": self.decision_note,
            "approval_latency_ms": self.approval_latency_ms,
            "outcome": self.outcome.value,
            "duration_ms": self.duration_ms,
            "error": dict(self.error) if self.error else None,
            "kill_switch": dict(self.kill_switch) if self.kill_switch else None,
            "labels": dict(self.labels) if self.labels else None,
        }
        return record

    def with_chain(self, *, seq: int, prev_hash: str, digest: str) -> AuditEvent:
        """Return a copy carrying the chain fields the sink computed."""
        return dataclasses.replace(self, seq=seq, prev_hash=prev_hash, hash=digest)

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> AuditEvent:
        """Rebuild an event from a parsed log line, for reporting.

        Unknown stages/outcomes from a newer writer degrade to sensible defaults
        rather than raising, so an old reader can still summarise a new log.
        """
        def _enum(enum_cls: Any, value: Any, fallback: Any) -> Any:
            try:
                return enum_cls(value)
            except (ValueError, TypeError):
                return fallback

        degraded = record.get("degraded_from")
        return cls(
            capability=str(record.get("capability", "<unknown>")),
            stage=_enum(Stage, record.get("stage"), Stage.OBSERVE),
            declared_stage=_enum(Stage, record.get("declared_stage"), Stage.OBSERVE),
            outcome=_enum(Outcome, record.get("outcome"), Outcome.ERROR),
            decision=_enum(Decision, record.get("decision"), Decision.NOT_REQUIRED),
            correlation_id=str(record.get("correlation_id", "")),
            event_id=str(record.get("event_id", "")),
            timestamp=str(record.get("timestamp", "")),
            risk_tier=_enum(RiskTier, record.get("risk_tier"), RiskTier.MODERATE),
            arguments=record.get("arguments") or {},
            effect=str(record.get("effect", "")),
            actor=record.get("actor"),
            decision_actor=record.get("decision_actor"),
            decision_note=record.get("decision_note"),
            approval_latency_ms=record.get("approval_latency_ms"),
            duration_ms=record.get("duration_ms"),
            parent_event_id=record.get("parent_event_id"),
            depth=int(record.get("depth") or 0),
            error=record.get("error"),
            kill_switch=record.get("kill_switch"),
            degraded_from=_enum(Stage, degraded, None) if degraded else None,
            labels=record.get("labels"),
            seq=record.get("seq"),
            prev_hash=record.get("prev_hash"),
            hash=record.get("hash"),
        )


def chain_digest(record: Mapping[str, Any], prev_hash: str) -> str:
    """Compute a record's chain hash. ``record`` must not contain ``hash``."""
    payload = {k: v for k, v in record.items() if k != "hash"}
    return hashlib.sha256((prev_hash + canonical_json(payload)).encode("utf-8")).hexdigest()


@runtime_checkable
class AuditSink(Protocol):
    """Destination for audit events.

    Implement this to ship events to a SIEM, a database, or a message bus. The
    gate calls :meth:`preflight` before it lets anything execute, so a sink that
    knows it cannot accept writes gets to stop an action before it happens rather
    than after.
    """

    def emit(self, event: AuditEvent) -> AuditEvent:
        """Durably record ``event``; return it with chain fields populated.

        Raises:
            AuditWriteError: if the record could not be durably recorded.
        """
        ...

    def preflight(self) -> None:
        """Raise :class:`~stagegate.errors.AuditWriteError` if writes cannot succeed."""
        ...

    def close(self) -> None:
        """Release resources. Must be idempotent."""
        ...


class JsonlAuditLog:
    """Append-only JSONL sink with hash chaining, safe for concurrent writers.

    Args:
        path: Log file. Parent directories are created. Opened ``O_APPEND`` with
            mode ``0o600``.
        fsync: ``fsync`` after every record. Correct if you need the record to
            survive a machine losing power between the write and the action;
            costly, so it is opt-in and documented rather than silently on.
        on_corrupt: What to do when an existing log fails its opening integrity
            check. ``"raise"`` (default) refuses to start -- fail closed, an
            operator looks. ``"seal"`` renames the suspect file aside and starts a
            fresh chain whose first record names the sealed file, so an agent can
            restart without an operator but the break is still in the record.

    Raises:
        AuditLogCorruption: existing log fails verification and ``on_corrupt="raise"``.
        AuditWriteError: the path cannot be opened for appending.

    Note:
        Each record is written with a single ``os.write`` of the whole line. On
        Linux and macOS an ``O_APPEND`` write below ``PIPE_BUF`` is atomic with
        respect to other appenders, so several processes may share one log without
        interleaving. The hash chain, however, assumes a single writer: two
        processes will both chain onto the same ``prev_hash`` and
        :func:`verify_chain` will flag it. Give each process its own file, or use
        a sink with a real serialisation point.
    """

    def __init__(
        self,
        path: str | os.PathLike[str],
        *,
        fsync: bool = False,
        on_corrupt: Literal["raise", "seal"] = "raise",
    ) -> None:
        self.path = Path(path)
        self.fsync = fsync
        self._lock = threading.Lock()
        self._closed = False
        self._seq = 0
        self._prev_hash = GENESIS_HASH
        self._sealed_from: str | None = None

        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._recover(on_corrupt)
        try:
            self._fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        except OSError as exc:
            raise AuditWriteError(f"cannot open audit log {self.path}: {exc}") from exc

    def _recover(self, on_corrupt: Literal["raise", "seal"]) -> None:
        """Restore chain position from an existing log, repairing a torn tail."""
        if not self.path.exists():
            return
        try:
            size = self.path.stat().st_size
        except OSError as exc:
            raise AuditWriteError(f"cannot stat audit log {self.path}: {exc}") from exc
        if size == 0:
            return

        self._truncate_torn_tail()
        last = _read_last_line(self.path)
        if last is None:
            return

        problem: str | None = None
        try:
            record = json.loads(last)
            if not isinstance(record, dict):
                raise ValueError("record is not an object")
            stored = record.get("hash")
            expected = chain_digest(record, str(record.get("prev_hash", GENESIS_HASH)))
            if stored != expected:
                problem = "hash of the final record does not match its contents"
            else:
                self._seq = int(record.get("seq", 0))
                self._prev_hash = str(stored)
        except (ValueError, TypeError) as exc:
            problem = f"final record is not valid JSON: {exc}"

        if problem is None:
            return
        if on_corrupt == "raise":
            raise AuditLogCorruption(
                f"{self.path}: {problem}. Run `stagegate verify` to locate the break, "
                f"or construct the sink with on_corrupt='seal' to quarantine it.",
                path=str(self.path),
            )
        sealed = self.path.with_name(
            f"{self.path.name}.corrupt-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
        )
        self.path.rename(sealed)
        self._sealed_from = sealed.name
        self._seq = 0
        self._prev_hash = GENESIS_HASH

    def _truncate_torn_tail(self) -> None:
        """Drop a trailing partial line left by a crash mid-write.

        A line without its terminating newline was never a committed record, so
        removing it does not violate append-only -- it is the same recovery a
        write-ahead log performs at startup.
        """
        try:
            with open(self.path, "rb+") as handle:
                handle.seek(0, os.SEEK_END)
                size = handle.tell()
                if size == 0:
                    return
                handle.seek(size - 1)
                if handle.read(1) == b"\n":
                    return
                handle.seek(0, os.SEEK_END)
                keep = size
                chunk = 8192
                while keep > 0:
                    start = max(0, keep - chunk)
                    handle.seek(start)
                    data = handle.read(keep - start)
                    index = data.rfind(b"\n")
                    if index != -1:
                        handle.truncate(start + index + 1)
                        return
                    keep = start
                handle.truncate(0)
        except OSError as exc:
            raise AuditWriteError(f"cannot repair audit log {self.path}: {exc}") from exc

    def emit(self, event: AuditEvent) -> AuditEvent:
        """Append ``event`` and return it with ``seq``, ``prev_hash`` and ``hash`` set."""
        if self._closed:
            raise AuditWriteError("audit log is closed")
        with self._lock:
            seq = self._seq + 1
            prev = self._prev_hash
            record = event.to_record()
            record["seq"] = seq
            record["prev_hash"] = prev
            if self._sealed_from is not None:
                record["chain_reset_from"] = self._sealed_from
                self._sealed_from = None
            digest = chain_digest(record, prev)
            record["hash"] = digest
            line = (canonical_json(record) + "\n").encode("utf-8")
            try:
                written = os.write(self._fd, line)
                if written != len(line):  # pragma: no cover - short write on a regular file
                    raise AuditWriteError(
                        f"short write to {self.path}: {written} of {len(line)} bytes"
                    )
                if self.fsync:
                    os.fsync(self._fd)
            except OSError as exc:
                raise AuditWriteError(f"cannot append to {self.path}: {exc}") from exc
            self._seq = seq
            self._prev_hash = digest
        return event.with_chain(seq=seq, prev_hash=prev, digest=digest)

    def preflight(self) -> None:
        """Confirm the descriptor is still writable before an action is allowed."""
        if self._closed:
            raise AuditWriteError("audit log is closed")
        try:
            os.fstat(self._fd)
        except OSError as exc:
            raise AuditWriteError(f"audit log {self.path} is not writable: {exc}") from exc

    def close(self) -> None:
        """Close the descriptor. Idempotent."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            with contextlib.suppress(OSError):  # descriptor already gone
                os.close(self._fd)

    def __enter__(self) -> JsonlAuditLog:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def verify(self) -> ChainVerification:
        """Verify this log's chain end to end."""
        return verify_chain(self.path)


class InMemoryAuditLog:
    """Chain-consistent sink that keeps records in a list. For tests and dry runs."""

    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []
        self.events: list[AuditEvent] = []
        self._lock = threading.Lock()
        self._prev_hash = GENESIS_HASH
        self._closed = False
        self.fail_next: Exception | None = None
        """Set to an exception to simulate a sink failure on the next emit."""

    def emit(self, event: AuditEvent) -> AuditEvent:
        if self.fail_next is not None:
            error, self.fail_next = self.fail_next, None
            raise error
        with self._lock:
            seq = len(self.records) + 1
            prev = self._prev_hash
            record = event.to_record()
            record["seq"] = seq
            record["prev_hash"] = prev
            digest = chain_digest(record, prev)
            record["hash"] = digest
            self._prev_hash = digest
            chained = event.with_chain(seq=seq, prev_hash=prev, digest=digest)
            self.records.append(record)
            self.events.append(chained)
        return chained

    def preflight(self) -> None:
        if self._closed:
            raise AuditWriteError("audit log is closed")

    def close(self) -> None:
        self._closed = True


class StreamAuditSink:
    """Last-resort sink that writes JSONL to a stream (``stderr`` by default).

    Used as the gate's fallback when the primary sink fails: an event that cannot
    reach the log of record should still reach *something* an operator will see.
    Chains its own records so the fallback stream is itself verifiable.
    """

    def __init__(self, stream: TextIO | None = None) -> None:
        self._stream = stream if stream is not None else sys.stderr
        self._lock = threading.Lock()
        self._prev_hash = GENESIS_HASH
        self._seq = 0

    def emit(self, event: AuditEvent) -> AuditEvent:
        with self._lock:
            self._seq += 1
            prev = self._prev_hash
            record = event.to_record()
            record["seq"] = self._seq
            record["prev_hash"] = prev
            digest = chain_digest(record, prev)
            record["hash"] = digest
            self._prev_hash = digest
            try:
                self._stream.write(canonical_json(record) + "\n")
                self._stream.flush()
            except (OSError, ValueError) as exc:
                raise AuditWriteError(f"cannot write to fallback stream: {exc}") from exc
        return event.with_chain(seq=self._seq, prev_hash=prev, digest=digest)

    def preflight(self) -> None:
        if getattr(self._stream, "closed", False):
            raise AuditWriteError("fallback stream is closed")

    def close(self) -> None:
        return None


class MultiSink:
    """Fan an event out to several sinks.

    The first sink is the log of record: its chain fields are returned and its
    failures propagate. Later sinks are best-effort mirrors whose failures are
    collected in :attr:`errors` rather than allowed to stop an agent, because a
    flaky SIEM should not be able to halt production.

    Args:
        *sinks: The log of record first, then any mirrors.
        history: How many mirror failures to retain. Bounded: a mirror that has
            been down for a week would otherwise accumulate one exception object
            per audited call for the life of the process.
    """

    def __init__(self, *sinks: AuditSink, history: int = 256) -> None:
        if not sinks:
            raise ValueError("MultiSink needs at least one sink")
        self.sinks = sinks
        self.errors: deque[tuple[AuditSink, Exception]] = deque(maxlen=max(1, history))
        """Most recent ``history`` mirror failures, oldest dropped first."""

    def emit(self, event: AuditEvent) -> AuditEvent:
        primary = self.sinks[0].emit(event)
        for sink in self.sinks[1:]:
            try:
                sink.emit(event)
            except Exception as exc:  # noqa: BLE001 - mirrors must never break the caller
                self.errors.append((sink, exc))
        return primary

    def preflight(self) -> None:
        self.sinks[0].preflight()

    def close(self) -> None:
        for sink in self.sinks:
            # Closing is best effort: one sink refusing to close must not leave
            # the others open.
            with contextlib.suppress(Exception):
                sink.close()


@dataclass(frozen=True)
class ChainVerification:
    """Result of checking a log's hash chain.

    Attributes:
        ok: True when every record verified and the sequence is unbroken.
        count: Records read.
        first_bad_seq: Sequence number where verification first failed.
        reason: Human-readable explanation of the first failure.
        head_hash: Hash of the last verified record, suitable for anchoring
            externally (publish it, and any later edit to the log is detectable
            even by someone who can rewrite the whole file).
    """

    ok: bool
    count: int
    first_bad_seq: int | None = None
    reason: str | None = None
    head_hash: str | None = None


def verify_chain(path: str | os.PathLike[str]) -> ChainVerification:
    """Verify an audit log's hash chain from the genesis record.

    Detects edited fields, deleted records, reordering, and re-chained forgeries
    that did not recompute every subsequent hash.

    Returns:
        A :class:`ChainVerification`. Verification failure is a *result*, not an
        exception -- callers routinely want to report on a broken log rather than
        crash on one. An unreadable file still raises ``OSError``.
    """
    prev = GENESIS_HASH
    expected_seq = 0
    count = 0
    try:
        with open(path, encoding="utf-8") as handle:
            for line_no, raw_line in enumerate(handle, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                expected_seq += 1
                try:
                    record = json.loads(line)
                except ValueError as exc:
                    return ChainVerification(
                        False, count, expected_seq, f"line {line_no} is not valid JSON: {exc}", prev
                    )
                if not isinstance(record, dict):
                    return ChainVerification(
                        False, count, expected_seq, f"line {line_no} is not a JSON object", prev
                    )
                seq = record.get("seq")
                if seq != expected_seq:
                    return ChainVerification(
                        False, count, expected_seq,
                        f"line {line_no}: expected seq {expected_seq}, found {seq!r} "
                        f"(a record was removed, reordered, or inserted)", prev,
                    )
                if record.get("prev_hash") != prev:
                    return ChainVerification(
                        False, count, expected_seq,
                        f"line {line_no}: prev_hash does not match the previous "
                        f"record's hash", prev,
                    )
                digest = chain_digest(record, prev)
                if record.get("hash") != digest:
                    return ChainVerification(
                        False, count, expected_seq,
                        f"line {line_no}: contents do not match the recorded hash "
                        f"(record was modified after it was written)", prev,
                    )
                prev = digest
                count += 1
    except FileNotFoundError:
        return ChainVerification(False, 0, None, f"no such audit log: {path}", None)
    return ChainVerification(True, count, None, None, prev if count else None)


def read_events(path: str | os.PathLike[str], *, strict: bool = False) -> Iterator[AuditEvent]:
    """Yield events from a JSONL audit log.

    Args:
        path: Log to read.
        strict: Raise on a malformed line instead of skipping it. Reporting
            defaults to lenient so a single bad line cannot deny you the report;
            :func:`verify_chain` is where you go for integrity.

    Raises:
        AuditLogCorruption: on a malformed line when ``strict`` is set.
    """
    with open(path, encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError("record is not an object")
            except ValueError as exc:
                if strict:
                    raise AuditLogCorruption(
                        f"{path}: line {line_no} is malformed: {exc}", path=str(path)
                    ) from exc
                continue
            yield AuditEvent.from_record(record)


def _read_last_line(path: Path) -> str | None:
    """Return the final complete line, seeking backwards rather than reading all."""
    try:
        with open(path, "rb") as handle:
            handle.seek(0, os.SEEK_END)
            end = handle.tell()
            if end == 0:
                return None
            if end >= 1:
                handle.seek(end - 1)
                if handle.read(1) == b"\n":
                    end -= 1
            if end == 0:
                return None
            chunk = 8192
            pos = end
            buffer = b""
            while pos > 0:
                start = max(0, pos - chunk)
                handle.seek(start)
                buffer = handle.read(pos - start) + buffer
                index = buffer.rfind(b"\n")
                if index != -1:
                    return buffer[index + 1 :].decode("utf-8", "replace").strip() or None
                pos = start
            return buffer.decode("utf-8", "replace").strip() or None
    except OSError:
        return None
