"""Shared fixtures.

Every test runs with the StageGate environment variables cleared. Without this,
a developer who happens to have ``STAGEGATE_KILL`` exported in their shell gets a
test suite that fails in ways that look like library bugs.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from pathlib import Path

import pytest

from stagegate import (
    InMemoryAuditLog,
    KillSwitch,
    RiskTier,
    Stage,
    StageGate,
    StaticApprovalHandler,
)
from stagegate.approval import ApprovalRequest

STAGEGATE_ENV = (
    "STAGEGATE_KILL",
    "STAGEGATE_KILL_FILE",
    "STAGEGATE_REDACTION_SALT",
    "STAGEGATE_ACTOR",
)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Isolate every test from the developer's shell environment."""
    for name in STAGEGATE_ENV:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def sink() -> InMemoryAuditLog:
    return InMemoryAuditLog()


@pytest.fixture
def kill_file(tmp_path: Path) -> Path:
    """Path to a sentinel that does not exist yet."""
    return tmp_path / "kill"


@pytest.fixture
def gate(sink: InMemoryAuditLog, kill_file: Path) -> StageGate:
    """A gate wired to in-memory everything, with approvals granted by default."""
    return StageGate(
        audit=sink,
        kill_switch=KillSwitch(path=kill_file),
        approval=StaticApprovalHandler(True, actor="approver@example.com"),
        approval_timeout=1.0,
        actor="agent@example.com",
    )


@pytest.fixture
def calls() -> list[tuple[str, ...]]:
    """Records every real invocation, so a test can assert nothing ran."""
    return []


def make_request(
    *,
    capability: str = "demo.cap",
    risk: RiskTier = RiskTier.MODERATE,
    timeout: float = 1.0,
) -> ApprovalRequest:
    """Build an approval request for handler tests."""
    return ApprovalRequest(
        request_id="apr-test",
        capability=capability,
        effect="do the thing",
        arguments={"target": "widget-1"},
        risk_tier=risk,
        stage=Stage.SUGGEST,
        correlation_id="run-test",
        event_id="evt-test",
        requested_at=time.monotonic(),
        timeout_s=timeout,
    )


@pytest.fixture
def approval_request() -> Iterator[ApprovalRequest]:
    yield make_request()
