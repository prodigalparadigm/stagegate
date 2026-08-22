"""Correlation identifiers that tie a chain of calls to one agent run.

An agent run is a single logical episode: one user request, one scheduled job,
one incoming webhook. Everything that happens inside it -- including capabilities
that call other capabilities -- carries the same ``correlation_id``, and nested
calls additionally carry the ``event_id`` of their caller as ``parent_event_id``.
That pair is what makes a post-hoc reconstruction possible: the correlation id
gathers the run, the parent pointer orders it into a tree.

State lives in :mod:`contextvars`, so concurrent runs in the same process
(asyncio tasks, or threads started with a copied context) do not bleed into each
other. A bare ``threading.Thread`` starts from an empty context by default; pass
the correlation id explicitly across such a boundary.
"""

from __future__ import annotations

import contextvars
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

__all__ = ["RunContext", "agent_run", "current_run", "current_correlation_id", "new_id"]

_CURRENT_RUN: contextvars.ContextVar["RunContext | None"] = contextvars.ContextVar(
    "stagegate_current_run", default=None
)


def new_id(prefix: str = "") -> str:
    """Return a fresh, collision-resistant identifier.

    Args:
        prefix: Optional short prefix, joined with ``-``. Useful for eyeballing
            a log (``run-...``, ``evt-...``) without parsing it.
    """
    token = uuid.uuid4().hex
    return f"{prefix}-{token}" if prefix else token


@dataclass(frozen=True)
class RunContext:
    """Identity of the agent run currently in scope.

    Attributes:
        correlation_id: Stable id for the whole run.
        actor: Who or what the run is acting as. Recorded on every event.
        parent_event_id: The enclosing capability's event id, or ``None`` at the
            top level. Set by the gate as calls nest; not usually set by hand.
        depth: Nesting depth, 0 at the top level.
        labels: Free-form key/value tags copied onto every event in the run
            (deployment, tenant, ticket, model version...).
    """

    correlation_id: str
    actor: str | None = None
    parent_event_id: str | None = None
    depth: int = 0
    labels: dict[str, str] | None = None

    def child(self, event_id: str) -> RunContext:
        """Return the context that capabilities invoked *inside* ``event_id`` see."""
        return RunContext(
            correlation_id=self.correlation_id,
            actor=self.actor,
            parent_event_id=event_id,
            depth=self.depth + 1,
            labels=self.labels,
        )


@contextmanager
def agent_run(
    correlation_id: str | None = None,
    *,
    actor: str | None = None,
    labels: dict[str, str] | None = None,
    nest: bool = False,
) -> Iterator[RunContext]:
    """Open an agent run, so every capability call inside shares a correlation id.

    Args:
        correlation_id: Reuse an upstream id (a request id, a queue message id)
            to stitch StageGate's audit trail to the rest of your tracing. A new
            one is minted when omitted.
        actor: Identity the run acts as, recorded on every event in it.
        labels: Tags copied onto every event in the run.
        nest: By default, opening a run inside an existing run is a no-op that
            yields the outer run, because a run is meant to be the outermost
            boundary and silently starting a second one would split the audit
            trail in half. Pass ``nest=True`` when you genuinely want a distinct
            sub-run (a fan-out worker, say).

    Yields:
        The :class:`RunContext` now in scope.

    Example:
        >>> with agent_run(actor="support-bot@example.com") as run:
        ...     _ = run.correlation_id  # every call below shares it
    """
    existing = _CURRENT_RUN.get()
    if existing is not None and not nest and correlation_id is None:
        yield existing
        return

    context = RunContext(
        correlation_id=correlation_id or new_id("run"),
        actor=actor if actor is not None else (existing.actor if existing else None),
        parent_event_id=existing.parent_event_id if (existing and nest) else None,
        depth=(existing.depth + 1) if (existing and nest) else 0,
        labels=labels if labels is not None else (existing.labels if existing else None),
    )
    token = _CURRENT_RUN.set(context)
    try:
        yield context
    finally:
        _CURRENT_RUN.reset(token)


def current_run() -> RunContext | None:
    """Return the run in scope, or ``None`` outside any :func:`agent_run`."""
    return _CURRENT_RUN.get()


def current_correlation_id() -> str | None:
    """Return the correlation id in scope, or ``None`` outside any run."""
    run = _CURRENT_RUN.get()
    return run.correlation_id if run else None


@contextmanager
def _scoped(context: RunContext) -> Iterator[None]:
    """Internal: install ``context`` for the duration of a nested capability call."""
    token = _CURRENT_RUN.set(context)
    try:
        yield
    finally:
        _CURRENT_RUN.reset(token)
