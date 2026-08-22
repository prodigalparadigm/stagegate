"""Documentation and the shipped example are part of the deliverable, so they are tested."""

from __future__ import annotations

import doctest
import importlib
import io
import re
import subprocess
import sys
import tomllib
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
        check=False,
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
        check=False,
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
        capture_output=True, text=True, timeout=60, cwd=ROOT, check=False,
    )
    assert result.returncode == 0
    assert "report" in result.stdout and "verify" in result.stdout


def test_the_shipped_implementations_satisfy_the_protocols_they_advertise() -> None:
    """The README tells people to implement these protocols. The built-ins must fit them."""
    import stagegate

    for sink in (
        stagegate.InMemoryAuditLog(),
        stagegate.StreamAuditSink(io.StringIO()),
        stagegate.MultiSink(stagegate.InMemoryAuditLog()),
    ):
        assert isinstance(sink, stagegate.AuditSink), type(sink).__name__
        # runtime_checkable only checks names exist, so call the contract too.
        sink.preflight()
        sink.close()

    for handler in (
        stagegate.StaticApprovalHandler(True),
        stagegate.CLIApprovalHandler(),
        stagegate.QueueApprovalHandler(),
    ):
        assert isinstance(handler, stagegate.ApprovalHandler), type(handler).__name__

    assert isinstance(stagegate.RedactionPolicy(), stagegate.Redactor)
    def plain_function_redactor(arguments):
        return dict(arguments)

    assert isinstance(plain_function_redactor, stagegate.Redactor)


def test_the_package_version_matches_the_packaging_metadata() -> None:
    """A version that drifts from pyproject.toml makes every release note a lie."""
    import stagegate

    declared = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert stagegate.__version__ == declared["project"]["version"]


def test_the_readme_quickstart_commands_are_the_ones_that_exist() -> None:
    """Guard the two shell commands the README promises a reader can copy."""
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert 'uv pip install -e ".[dev]"' in readme
    assert "uv venv" in readme, "installing without creating a venv first fails under uv"

    declared = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert declared["project"]["scripts"]["stagegate"] == "stagegate.__main__:main"
    for command in ("stagegate report", "stagegate verify"):
        assert command in readme
        subcommand = command.split()[1]
        result = subprocess.run(
            [sys.executable, "-m", "stagegate", subcommand, "--help"],
            capture_output=True, text=True, timeout=60, cwd=ROOT, check=False,
        )
        assert result.returncode == 0, result.stderr


def test_the_declared_test_count_in_the_readme_is_true() -> None:
    """A README that overstates its own test suite is the cheapest thing to catch.

    Collection runs in a subprocess rather than reading this session's own count,
    so the check is the same whether you run the whole suite or just this file.
    """
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    claimed = re.search(r"#\s*(\d+) tests, no network", readme)
    assert claimed is not None, "the quickstart should say how many tests there are"

    result = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider"],
        capture_output=True, text=True, timeout=180, cwd=ROOT, check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    collected = _collected_count(result.stdout)
    assert collected is not None, result.stdout[-2000:]
    assert int(claimed.group(1)) == collected, (
        f"README claims {claimed.group(1)} tests; collection found {collected}"
    )


def _collected_count(stdout: str) -> int | None:
    """Total collected tests from ``pytest --collect-only -q``.

    pytest 9 prints a per-file tally; pytest 8 prints node ids and a total. The
    dev dependency allows both, so read both rather than pinning a version to
    keep one regex happy.
    """
    per_file = re.findall(r"^\S+\.py: (\d+)$", stdout, re.MULTILINE)
    if per_file:
        return sum(int(n) for n in per_file)
    total = re.search(r"^(\d+) tests? collected", stdout, re.MULTILINE)
    return int(total.group(1)) if total else None
