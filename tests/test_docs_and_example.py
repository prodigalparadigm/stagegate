"""Documentation and the shipped example are part of the deliverable, so they are tested."""

from __future__ import annotations

import doctest
import importlib
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

MODULES = [
    "stagegate",
    "stagegate.approval",
    "stagegate.audit",
    "stagegate.correlation",
    "stagegate.errors",
    "stagegate.gate",
    "stagegate.killswitch",
    "stagegate.policy",
    "stagegate.redaction",
    "stagegate.report",
    "stagegate.stages",
]


@pytest.mark.parametrize("module_name", MODULES)
def test_docstring_examples_actually_work(module_name: str) -> None:
    module = importlib.import_module(module_name)
    result = doctest.testmod(module, verbose=False, optionflags=doctest.ELLIPSIS)
    assert result.failed == 0, f"{result.failed} doctest failure(s) in {module_name}"


def test_every_public_name_is_importable() -> None:
    import stagegate

    for name in stagegate.__all__:
        assert hasattr(stagegate, name), f"{name} is exported but missing"


def test_the_example_runs_end_to_end() -> None:
    """The README tells a reader to run this. It has to work."""
    result = subprocess.run(
        [sys.executable, str(ROOT / "examples" / "triage_agent.py"), "--report"],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=ROOT,
    )
    assert result.returncode == 0, result.stderr
    output = result.stdout

    # Policy holds tickets.transition at OBSERVE even though the code declares ACT.
    assert "tickets.transition   act       observe" in output
    # The risk ceiling caps the critical capability at SUGGEST.
    assert "oncall.page          act       suggest" in output
    # Shadow mode really did not change the ticket.
    assert "'T-1001': 'open'" in output
    assert "audit chain: ok=True" in output
    assert "# Agent dry-run report" in output


def test_the_example_degrades_under_a_tripped_kill_switch() -> None:
    result = subprocess.run(
        [sys.executable, str(ROOT / "examples" / "triage_agent.py"), "--tripped"],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=ROOT,
    )
    assert result.returncode == 0, "a tripped switch must not crash the agent"
    assert "blocked" in result.stdout
    assert "comments: []" in result.stdout
    assert "pages   : []" in result.stdout


def test_the_readme_quickstart_imports_resolve() -> None:
    """Cheap guard against the README drifting from the API."""
    import stagegate

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for name in (
        "StageGate", "Stage", "RiskTier", "JsonlAuditLog", "agent_run",
        "StagePolicy", "QueueApprovalHandler",
    ):
        assert name in readme
        assert hasattr(stagegate, name)


def test_the_cli_entry_point_is_wired_up() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "stagegate", "--help"],
        capture_output=True, text=True, timeout=60, cwd=ROOT,
    )
    assert result.returncode == 0
    assert "report" in result.stdout and "verify" in result.stdout
