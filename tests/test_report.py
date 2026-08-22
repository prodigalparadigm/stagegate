"""The dry-run report and the CLI that produces it."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from stagegate import (
    REDACTED,
    JsonlAuditLog,
    KillSwitch,
    RiskTier,
    Stage,
    StageGate,
    StagePolicy,
    StaticApprovalHandler,
    agent_run,
    build_report,
    report_from_log,
)
from stagegate.__main__ import main


@pytest.fixture
def shadow_log(tmp_path: Path, kill_file: Path) -> Path:
    """A log from a realistic shadow run: observe, approvals, a block, an error."""
    path = tmp_path / "audit.jsonl"
    log = JsonlAuditLog(path)
    gate = StageGate(
        audit=log,
        kill_switch=KillSwitch(path=kill_file),
        approval=StaticApprovalHandler(True, actor="ops@example.com"),
        policy=StagePolicy(overrides={"tickets.comment": Stage.SUGGEST}),
    )

    @gate.capability(
        "tickets.transition",
        stage=Stage.OBSERVE,
        risk=RiskTier.HIGH,
        describe="Move {ticket_id} to {status}",
    )
    def transition(ticket_id: str, status: str, api_token: str = "sk-abcdefghijklmnopqrst") -> None:
        raise AssertionError("shadow mode must not execute")

    @gate.capability("tickets.comment", stage=Stage.OBSERVE, describe="Comment on {ticket_id}")
    def comment(ticket_id: str) -> str:
        return "ok"

    @gate.capability("tickets.flaky", stage=Stage.ACT, propagate_errors=False)
    def flaky() -> None:
        raise ConnectionError("upstream 503")

    for index in range(3):
        with agent_run(f"run-{index}", actor="bot@example.com"):
            transition(f"T-100{index}", "in_progress")
            transition(f"T-100{index}", "done")
            comment(f"T-100{index}")

    with agent_run("run-blocked"):
        kill_file.write_text("incident 4412\n")
        comment("T-9999")
        kill_file.unlink()
        flaky()

    log.close()
    return path


def test_the_report_counts_what_happened(shadow_log: Path) -> None:
    report = report_from_log(shadow_log)
    # 3 runs x (2 shadow transitions + 1 approved comment) = 9,
    # plus a comment blocked by the kill switch and one failed execution.
    assert report.total_events == 11
    assert len(report.runs) == 4
    assert report.outcomes["recorded"] == 6
    assert report.outcomes["executed"] == 3
    assert report.outcomes["blocked"] == 1
    assert report.outcomes["failed"] == 1
    assert sum(report.outcomes.values()) == report.total_events


def test_the_report_verifies_the_chain_it_read(shadow_log: Path) -> None:
    report = report_from_log(shadow_log)
    assert report.verification is not None
    assert report.verification.ok is True
    assert report.verification.head_hash


def test_a_tampered_log_is_reported_as_tampered_not_hidden(shadow_log: Path) -> None:
    rows = shadow_log.read_text().splitlines()
    row = json.loads(rows[1])
    row["capability"] = "something-else"
    rows[1] = json.dumps(row)
    shadow_log.write_text("\n".join(rows) + "\n")

    report = report_from_log(shadow_log)
    assert report.verification.ok is False
    assert "**FAILED at record 2**" in report.to_markdown()


def test_intended_effects_are_grouped_with_counts(shadow_log: Path) -> None:
    report = report_from_log(shadow_log)
    transition = next(c for c in report.capabilities if c.name == "tickets.transition")
    effects = {group.effect: group.count for group in transition.effects}
    assert effects["Move T-1000 to in_progress"] == 1
    assert effects["Move T-1002 to done"] == 1
    assert transition.distinct_effects == 6
    assert transition.shadow_calls == 6


def test_example_arguments_in_the_report_are_the_redacted_ones(shadow_log: Path) -> None:
    """The report is a document that gets circulated. It must be safe to circulate."""
    report = report_from_log(shadow_log)
    transition = next(c for c in report.capabilities if c.name == "tickets.transition")
    assert transition.effects[0].example_arguments["api_token"] == REDACTED
    assert "sk-abcdefghijkl" not in report.to_markdown()
    assert "sk-abcdefghijkl" not in report.to_json()


def test_approvals_are_summarised(shadow_log: Path) -> None:
    report = report_from_log(shadow_log)
    comment = next(c for c in report.capabilities if c.name == "tickets.comment")
    assert comment.decisions["approved"] == 3
    assert comment.approval_rate == 1.0
    assert comment.approvers["ops@example.com"] == 3
    assert comment.approval_latencies_ms


def test_kill_switch_activations_get_their_own_section(shadow_log: Path) -> None:
    report = report_from_log(shadow_log)
    assert len(report.kill_switch_events) == 1
    assert "incident 4412" in report.kill_switch_events[0]["reason"]
    assert "## Kill-switch activations" in report.to_markdown()


def test_errors_are_surfaced_per_capability(shadow_log: Path) -> None:
    report = report_from_log(shadow_log)
    flaky = next(c for c in report.capabilities if c.name == "tickets.flaky")
    assert flaky.errors[0]["type"] == "ConnectionError"
    assert "upstream 503" in report.to_markdown()


def test_the_report_refuses_to_claim_a_shadow_call_would_have_worked(shadow_log: Path) -> None:
    """The most important line in the document."""
    report = report_from_log(shadow_log)
    transition = next(c for c in report.capabilities if c.name == "tickets.transition")
    checks = {check: status for check, status, _ in transition.readiness()}
    assert checks["execution evidence"] == "unknown"

    markdown = report.to_markdown()
    assert "proves" not in markdown.lower() or "intended" in markdown.lower()
    assert "never ran" in markdown
    assert "evidence, not a recommendation" in markdown


def test_high_risk_capabilities_are_flagged_for_a_named_decision(shadow_log: Path) -> None:
    report = report_from_log(shadow_log)
    transition = next(c for c in report.capabilities if c.name == "tickets.transition")
    checks = {check: status for check, status, _ in transition.readiness()}
    assert checks["risk tier"] == "attention"


def test_a_thin_sample_is_called_thin(shadow_log: Path) -> None:
    report = report_from_log(shadow_log)
    transition = next(c for c in report.capabilities if c.name == "tickets.transition")
    details = {check: detail for check, _, detail in transition.readiness()}
    assert "thin sample" in details["shadow volume"]


def test_blocked_calls_are_flagged(shadow_log: Path) -> None:
    report = report_from_log(shadow_log)
    comment = next(c for c in report.capabilities if c.name == "tickets.comment")
    checks = {check: status for check, status, _ in comment.readiness()}
    assert checks["guard activations"] == "attention"


def test_capabilities_are_ordered_by_risk_then_volume(shadow_log: Path) -> None:
    report = report_from_log(shadow_log)
    assert report.capabilities[0].name == "tickets.transition"


def test_the_markdown_renders_and_stays_a_table(shadow_log: Path) -> None:
    markdown = report_from_log(shadow_log).to_markdown()
    assert markdown.startswith("# Agent dry-run report")
    assert "| Capability | Risk | Declared |" in markdown
    assert "## Sign-off" in markdown
    assert "Report head hash:" in markdown


def test_a_pipe_in_an_effect_does_not_break_the_markdown_table(sink) -> None:
    gate = StageGate(audit=sink)

    @gate.capability("demo.cap", stage=Stage.OBSERVE, describe="run `a | b` on {target}")
    def cap(target: str) -> None: ...

    cap("widget")
    markdown = build_report(sink.events).to_markdown()
    effect_row = next(line for line in markdown.splitlines() if "a \\| b" in line)
    unescaped = re.split(r"(?<!\\)\|", effect_row)
    assert len(unescaped) == 4, "leading, effect, count, trailing - the literal pipe is escaped"
    assert unescaped[1].strip() == "run `a \\| b` on widget"


def test_the_json_view_is_serialisable_and_complete(shadow_log: Path) -> None:
    payload = json.loads(report_from_log(shadow_log).to_json())
    assert payload["schema"] == "stagegate.report/1"
    assert payload["integrity"]["verified"] is True
    assert payload["total_runs"] == 4
    names = {cap["capability"] for cap in payload["capabilities"]}
    assert names == {"tickets.transition", "tickets.comment", "tickets.flaky"}


def test_effect_groups_are_bounded_for_a_capability_with_unbounded_effects(sink) -> None:
    """An effect string embedding a unique id must not make the report unbounded."""
    gate = StageGate(audit=sink)

    @gate.capability("demo.cap", stage=Stage.OBSERVE, describe="handle {item_id}")
    def cap(item_id: str) -> None: ...

    for index in range(50):
        cap(f"item-{index}")

    report = build_report(sink.events, max_effect_groups=10)
    capability = report.capabilities[0]
    assert capability.distinct_effects == 11, "ten groups plus one overflow bucket"
    assert sum(group.count for group in capability.effects) == 50, "totals still add up"


def test_an_empty_log_produces_an_empty_but_valid_report(tmp_path: Path) -> None:
    path = tmp_path / "empty.jsonl"
    path.write_text("")
    report = report_from_log(path)
    assert report.total_events == 0
    assert report.capabilities == []
    assert report.to_markdown().startswith("# Agent dry-run report")
    json.loads(report.to_json())


# ------------------------------------------------------------------ the CLI


def test_cli_report_writes_markdown_to_stdout(shadow_log: Path, capsys) -> None:
    assert main(["report", str(shadow_log)]) == 0
    assert "# Agent dry-run report" in capsys.readouterr().out


def test_cli_report_writes_json_to_a_file(shadow_log: Path, tmp_path: Path) -> None:
    out = tmp_path / "report.json"
    assert main(["report", str(shadow_log), "--format", "json", "-o", str(out)]) == 0
    assert json.loads(out.read_text())["schema"] == "stagegate.report/1"


def test_cli_verify_succeeds_on_a_clean_log(shadow_log: Path, capsys) -> None:
    assert main(["verify", str(shadow_log)]) == 0
    assert "OK:" in capsys.readouterr().out


def test_cli_verify_exits_nonzero_on_a_broken_chain(shadow_log: Path, capsys) -> None:
    """This is the check that belongs in CI."""
    rows = shadow_log.read_text().splitlines()
    row = json.loads(rows[0])
    row["capability"] = "tickets.something_harmless"
    rows[0] = json.dumps(row)
    shadow_log.write_text("\n".join(rows) + "\n")

    assert main(["verify", str(shadow_log)]) == 1
    assert "FAILED at record 1" in capsys.readouterr().err


def test_cli_report_can_fail_on_tamper(shadow_log: Path, capsys) -> None:
    rows = shadow_log.read_text().splitlines()
    row = json.loads(rows[0])
    row["capability"] = "tickets.something_harmless"
    rows[0] = json.dumps(row)
    shadow_log.write_text("\n".join(rows) + "\n")

    assert main(["report", str(shadow_log), "--fail-on-tamper"]) == 1


def test_cli_reports_a_missing_file_as_a_usage_error(tmp_path: Path, capsys) -> None:
    assert main(["report", str(tmp_path / "nope.jsonl")]) == 2
    assert "cannot read" in capsys.readouterr().err


def test_cli_verify_quiet_prints_nothing(shadow_log: Path, capsys) -> None:
    assert main(["verify", str(shadow_log), "--quiet"]) == 0
    captured = capsys.readouterr()
    assert captured.out == "" and captured.err == ""
