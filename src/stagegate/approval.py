"""Approval handlers: how a human gets asked, and what happens when nobody answers.

``SUGGEST`` stage is only as good as the thing that asks the human. Handlers are
a protocol with two supplied implementations:

* :class:`CLIApprovalHandler` -- prompts on a terminal. For an operator running an
  agent interactively, and for the first week of any rollout.
* :class:`QueueApprovalHandler` -- parks the request and blocks the calling thread
  until some *other* thread resolves it. That other thread is your web request
  handler, your Slack callback, your ticket webhook. This is the shape that works
  in production, where the approver is not sitting at the process's stdin.

Every handler takes a timeout and every timeout fails closed. Silence is not
consent: an unanswered request is recorded as ``TIMED_OUT`` and nothing executes.
The same goes for a handler that raises -- an approval system that is down cannot
approve, so it denies.
"""

from __future__ import annotations

import contextlib
import os
import queue
import select
import sys
import threading
import time
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, TextIO, runtime_checkable

from .errors import ApprovalError
from .stages import RiskTier, Stage

__all__ = [
    "ApprovalRequest",
    "ApprovalResponse",
    "ApprovalHandler",
    "CLIApprovalHandler",
    "QueueApprovalHandler",
    "StaticApprovalHandler",
    "PendingApproval",
    "DEFAULT_TIMEOUT_SECONDS",
]

DEFAULT_TIMEOUT_SECONDS = 300.0
"""Five minutes. Long enough for a human to read and think, short enough that a
blocked agent thread is noticed the same shift."""

_APPROVE_WORDS = frozenset({"y", "yes", "approve", "approved", "ok", "go"})


@dataclass(frozen=True, slots=True)
class ApprovalRequest:
    """Everything a human needs to decide, and nothing they should not see.

    ``arguments`` is already redacted: an approver reads the same sanitised view
    the audit log keeps. If the redaction policy hides something an approver
    genuinely needs, that is a signal to narrow the policy for that capability,
    not to hand the approver raw arguments.
    """

    request_id: str
    capability: str
    effect: str
    arguments: Mapping[str, Any]
    risk_tier: RiskTier
    stage: Stage
    correlation_id: str
    event_id: str
    requested_at: float
    timeout_s: float
    actor: str | None = None
    labels: Mapping[str, str] | None = None

    @property
    def deadline(self) -> float:
        """``time.monotonic()`` value after which the request is dead."""
        return self.requested_at + self.timeout_s

    def remaining(self) -> float:
        """Seconds left before the request times out; never negative."""
        return max(0.0, self.deadline - time.monotonic())

    def summary(self) -> str:
        """One-line description for a notification or a log."""
        return f"[{self.risk_tier.value}] {self.capability}: {self.effect}"


@dataclass(frozen=True, slots=True)
class ApprovalResponse:
    """A human's decision.

    Attributes:
        approved: Whether execution may proceed.
        actor: Who decided. Recorded in the audit log; an approval without an
            identifiable approver is not much of a control, so handlers make a
            genuine effort to fill this in.
        note: Free-text justification, recorded alongside the decision.
    """

    approved: bool
    actor: str | None = None
    note: str | None = None
    decided_at: float = field(default_factory=time.time)


@runtime_checkable
class ApprovalHandler(Protocol):
    """Asks a human and returns their decision.

    Implementations must honour ``request.timeout_s`` and must not raise for a
    denial -- a denial is a normal return value. Raise
    :class:`~stagegate.errors.ApprovalError` only when no decision could be
    obtained at all; the gate treats that as a refusal.
    """

    def request_approval(self, request: ApprovalRequest) -> ApprovalResponse:
        """Obtain a decision for ``request`` within its timeout."""
        ...


class StaticApprovalHandler:
    """Always answers the same way. For tests, examples, and local development.

    Never wire this into anything that can act on a real system. It is here so
    that a test of stage routing does not have to fake a terminal.
    """

    def __init__(
        self, approved: bool, *, actor: str = "static-handler", note: str | None = None
    ) -> None:
        self.approved = approved
        self.actor = actor
        self.note = note
        self.requests: list[ApprovalRequest] = []

    def request_approval(self, request: ApprovalRequest) -> ApprovalResponse:
        self.requests.append(request)
        return ApprovalResponse(self.approved, actor=self.actor, note=self.note)


class CLIApprovalHandler:
    """Prompts on a terminal and reads a line, with a real timeout.

    Args:
        stream_in: Where to read the answer. Defaults to ``sys.stdin``.
        stream_out: Where to write the prompt. Defaults to ``sys.stderr``, so a
            prompt never contaminates a program's stdout.
        actor: Identity to record. Defaults to ``$STAGEGATE_ACTOR``, then the OS
            login name.
        confirm_at_or_above: Risk tier at or above which a bare "y" is not
            enough and the approver must retype the capability name. Muscle memory
            approves anything; typing ``billing.issue_refund`` does not happen by
            accident. Set to ``None`` to disable.

    Note:
        The timeout is enforced with :func:`select.select` on the input stream's
        file descriptor. When the stream has no usable descriptor -- a pipe on
        Windows, a ``StringIO`` in a test -- the read cannot be interrupted, so the
        handler reads and then checks whether the deadline passed while it waited.
        A late answer is discarded rather than honoured. That degrades the *user
        experience* of the timeout, never its safety property.
    """

    def __init__(
        self,
        *,
        stream_in: TextIO | None = None,
        stream_out: TextIO | None = None,
        actor: str | None = None,
        confirm_at_or_above: RiskTier | None = RiskTier.CRITICAL,
    ) -> None:
        self._in = stream_in
        self._out = stream_out
        self._actor = actor
        self.confirm_at_or_above = confirm_at_or_above

    @property
    def stream_in(self) -> TextIO:
        return self._in if self._in is not None else sys.stdin

    @property
    def stream_out(self) -> TextIO:
        return self._out if self._out is not None else sys.stderr

    def actor(self) -> str:
        """Best available identity for the person at the terminal."""
        if self._actor:
            return self._actor
        from_env = os.environ.get("STAGEGATE_ACTOR")
        if from_env:
            return from_env
        try:
            return os.getlogin()
        except OSError:
            return os.environ.get("USER") or os.environ.get("USERNAME") or "unknown"

    def request_approval(self, request: ApprovalRequest) -> ApprovalResponse:
        out = self.stream_out
        needs_typed = (
            self.confirm_at_or_above is not None
            and request.risk_tier.rank >= self.confirm_at_or_above.rank
        )
        prompt = (
            f"type the capability name {request.capability!r} to approve"
            if needs_typed
            else "approve? [y/N]"
        )
        try:
            out.write("\n" + self._render(request) + f"\n{prompt} ")
            out.flush()
        except (OSError, ValueError) as exc:
            raise ApprovalError(f"cannot write approval prompt: {exc}") from exc

        answer = self._read_line(request.remaining())
        if answer is None:
            self._say(out, "-> no answer before timeout; denied\n")
            raise TimeoutError("approval timed out")

        answer = answer.strip()
        approved = answer == request.capability if needs_typed else answer.lower() in _APPROVE_WORDS
        self._say(out, f"-> {'approved' if approved else 'denied'}\n")
        return ApprovalResponse(
            approved,
            actor=self.actor(),
            note=None if approved else f"declined at prompt (answer: {answer[:80]!r})",
        )

    def _render(self, request: ApprovalRequest) -> str:
        lines = [
            "=" * 68,
            f"  APPROVAL REQUIRED   risk={request.risk_tier.value}   run={request.correlation_id}",
            "=" * 68,
            f"  capability : {request.capability}",
            f"  effect     : {request.effect}",
        ]
        if request.actor:
            lines.append(f"  agent      : {request.actor}")
        if request.arguments:
            lines.append("  arguments  :")
            for key, value in request.arguments.items():
                lines.append(f"      {key} = {value!r}")
        lines.append(f"  expires in : {request.remaining():.0f}s")
        lines.append("=" * 68)
        return "\n".join(lines)

    @staticmethod
    def _say(out: TextIO, text: str) -> None:
        try:
            out.write(text)
            out.flush()
        except (OSError, ValueError):
            pass

    def _read_line(self, timeout: float) -> str | None:
        """Read one line, or return ``None`` if the deadline passed first."""
        stream = self.stream_in
        if timeout <= 0:
            return None
        deadline = time.monotonic() + timeout
        fd = _fileno(stream)
        if fd is not None:
            try:
                ready, _, _ = select.select([fd], [], [], timeout)
            except (OSError, ValueError):
                ready = [fd]  # cannot poll: fall through to a blocking read
            if not ready:
                return None
        try:
            line = stream.readline()
        except (OSError, ValueError) as exc:
            raise ApprovalError(f"cannot read approval response: {exc}") from exc
        if line == "":  # EOF: nobody is there. Fail closed.
            return None
        if time.monotonic() > deadline:
            return None
        return line


def _fileno(stream: TextIO) -> int | None:
    """Return a pollable descriptor for ``stream``, or ``None`` if it has none."""
    try:
        fd = stream.fileno()
    except Exception:  # noqa: BLE001 - io.UnsupportedOperation, OSError, AttributeError
        return None
    return fd if isinstance(fd, int) and fd >= 0 else None


@dataclass
class PendingApproval:
    """A request parked in a :class:`QueueApprovalHandler`, awaiting a decision."""

    request: ApprovalRequest
    event: threading.Event = field(default_factory=threading.Event)
    response: ApprovalResponse | None = None


class QueueApprovalHandler:
    """Parks requests for an out-of-band approver on another thread.

    The capability call blocks; a web handler, a Slack action callback, or an
    operator console calls :meth:`resolve` from a different thread and the call
    resumes. This is the production shape: the approver is a person with a
    browser, not a process with a stdin.

    Args:
        default_timeout: Used when a request carries none.
        on_submit: Called with each new request as it parks. Send the
            notification here. Exceptions are swallowed and recorded on
            :attr:`notify_errors` -- a broken pager must not become an approval.
        history: How many notification failures and un-consumed arrival hints to
            retain. Both are bounded because this object lives for the life of a
            long-running agent, and an unbounded diagnostic list is a slow leak:
            a pager that has been broken for a week must not be the reason the
            process runs out of memory.

    Example:
        >>> handler = QueueApprovalHandler()
        >>> # thread A (the agent) blocks in request_approval(...)
        >>> # thread B (your web handler):
        >>> # for pending in handler.pending():
        >>> #     handler.resolve(pending.request.request_id, True, actor="ops@example.com")

    Note:
        Every wait is bounded, and a request is removed from :meth:`pending` the
        moment it resolves or expires. A decision that arrives after the timeout
        is refused by :meth:`resolve`, which returns ``False``: the agent has
        already moved on, and honouring a late approval would execute an action
        whose approval the audit log records as never having arrived.
    """

    def __init__(
        self,
        *,
        default_timeout: float = DEFAULT_TIMEOUT_SECONDS,
        on_submit: Callable[[ApprovalRequest], None] | None = None,
        history: int = 256,
    ) -> None:
        self.default_timeout = default_timeout
        self.on_submit = on_submit
        self.notify_errors: deque[tuple[str, Exception]] = deque(maxlen=max(1, history))
        """Most recent ``history`` notification failures, oldest dropped first."""
        self._lock = threading.Lock()
        self._pending: dict[str, PendingApproval] = {}
        self._arrivals: queue.Queue[str] = queue.Queue(maxsize=max(1, history))

    def request_approval(self, request: ApprovalRequest) -> ApprovalResponse:
        timeout = request.timeout_s if request.timeout_s > 0 else self.default_timeout
        entry = PendingApproval(request=request)
        with self._lock:
            if request.request_id in self._pending:
                # Clobbering the existing entry would strand the thread already
                # waiting on it until its own timeout, and a later resolve would
                # be applied to whichever request happened to win the race.
                raise ApprovalError(
                    f"approval request id {request.request_id!r} is already pending; "
                    "request ids must be unique per in-flight request"
                )
            self._pending[request.request_id] = entry
        self._record_arrival(request.request_id)

        if self.on_submit is not None:
            try:
                self.on_submit(request)
            except Exception as exc:  # noqa: BLE001 - notification is best effort
                self.notify_errors.append((request.request_id, exc))

        answered = entry.event.wait(timeout=timeout)
        with self._lock:
            self._pending.pop(request.request_id, None)
        if not answered or entry.response is None:
            raise TimeoutError(f"no decision for {request.request_id} within {timeout:g}s")
        return entry.response

    def _record_arrival(self, request_id: str) -> None:
        """Hint to a polling worker that a request arrived, without blocking.

        :meth:`pending` is the source of truth; this queue only exists so a worker
        can wait rather than spin. When nobody is consuming it, the oldest hint is
        discarded instead of letting the queue grow for the life of the process.
        """
        while True:
            try:
                self._arrivals.put_nowait(request_id)
                return
            except queue.Full:
                with contextlib.suppress(queue.Empty):
                    self._arrivals.get_nowait()

    def pending(self) -> list[PendingApproval]:
        """Snapshot of requests currently awaiting a decision."""
        with self._lock:
            return list(self._pending.values())

    def next_request(self, timeout: float | None = None) -> ApprovalRequest | None:
        """Block until a request arrives, for a worker that polls rather than subscribes.

        Returns ``None`` on timeout, or when the arrival referred to a request
        that has since resolved or expired.
        """
        try:
            request_id = self._arrivals.get(timeout=timeout)
        except queue.Empty:
            return None
        with self._lock:
            entry = self._pending.get(request_id)
        return entry.request if entry else None

    def resolve(
        self,
        request_id: str,
        approved: bool,
        *,
        actor: str | None = None,
        note: str | None = None,
    ) -> bool:
        """Record a decision and wake the blocked capability call.

        Returns:
            ``True`` if the decision was delivered; ``False`` if the request is
            unknown, already decided, or expired.
        """
        with self._lock:
            entry = self._pending.get(request_id)
            if entry is None or entry.response is not None:
                return False
            entry.response = ApprovalResponse(approved, actor=actor, note=note)
        entry.event.set()
        return True

    def deny_all(self, *, actor: str | None = None, note: str = "bulk denial") -> int:
        """Deny every pending request. What an operator reaches for during an incident.

        Returns:
            How many requests were denied.
        """
        denied = 0
        for entry in self.pending():
            if self.resolve(entry.request.request_id, False, actor=actor, note=note):
                denied += 1
        return denied
