"""Kill-switch precedence, fail-closed behaviour, and degradation."""

from __future__ import annotations

from pathlib import Path

import pytest

from stagegate import (
    InMemoryAuditLog,
    KillSwitch,
    Outcome,
    RiskTier,
    Stage,
    StageGate,
    StaticApprovalHandler,
)

# ------------------------------------------------------------- precedence


def test_clear_when_nothing_is_set(kill_file: Path) -> None:
    state = KillSwitch(path=kill_file).check()
    assert state.tripped is False
    assert state.source == "clear"


@pytest.mark.parametrize(
    "value", ["1", "true", "TRUE", "yes", "on", " trip ", "tripped", "stop", "kill"]
)
def test_recognised_true_values_trip_via_env(
    kill_file: Path, monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv("STAGEGATE_KILL", value)
    state = KillSwitch(path=kill_file).check()
    assert state.tripped is True
    assert state.source == "env"


@pytest.mark.parametrize("value", ["0", "false", "no", "off", "", "clear"])
def test_recognised_false_values_abstain(
    kill_file: Path, monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv("STAGEGATE_KILL", value)
    assert KillSwitch(path=kill_file).check().tripped is False


@pytest.mark.parametrize("value", ["yes-please", "maybe", "2", "off?", "TrUe-ish"])
def test_unparseable_env_values_fail_closed(
    kill_file: Path, monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    """A switch that ignores what it does not understand is not a switch."""
    monkeypatch.setenv("STAGEGATE_KILL", value)
    state = KillSwitch(path=kill_file).check()
    assert state.tripped is True
    assert "not a recognised value" in (state.reason or "")


def test_file_sentinel_trips_and_its_first_line_becomes_the_reason(kill_file: Path) -> None:
    kill_file.write_text("incident 4412: paused by a.ochieng\nsecond line ignored\n")
    state = KillSwitch(path=kill_file).check()
    assert state.tripped is True
    assert state.source == "file"
    assert "incident 4412" in (state.reason or "")
    assert "second line" not in (state.reason or "")


def test_env_false_does_not_clear_a_file_sentinel(
    kill_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The whole point of the precedence rule: a shell cannot resume a stopped agent."""
    kill_file.write_text("stopped by the incident commander\n")
    monkeypatch.setenv("STAGEGATE_KILL", "0")

    state = KillSwitch(path=kill_file).check()

    assert state.tripped is True
    assert state.source == "file"


def test_env_true_wins_over_an_absent_sentinel(
    kill_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("STAGEGATE_KILL", "1")
    assert KillSwitch(path=kill_file).check().source == "env"


def test_sentinel_path_comes_from_the_environment_when_not_given(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sentinel = tmp_path / "from-env"
    sentinel.write_text("configured by ops\n")
    monkeypatch.setenv("STAGEGATE_KILL_FILE", str(sentinel))

    state = KillSwitch().check()

    assert state.tripped is True
    assert "configured by ops" in (state.reason or "")


def test_with_no_sentinel_configured_only_the_env_is_consulted() -> None:
    assert KillSwitch().check().tripped is False


# ------------------------------------------------------------ fail closed


def test_an_unstattable_sentinel_path_trips(
    kill_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    switch = KillSwitch(path=kill_file)

    def explode(self: Path) -> bool:
        raise PermissionError("EACCES")

    monkeypatch.setattr(Path, "exists", explode)
    state = switch.check()

    assert state.tripped is True
    assert state.source == "error"


def test_a_probe_that_raises_trips(kill_file: Path) -> None:
    def unreachable() -> tuple[bool, str | None]:
        raise TimeoutError("flag service unreachable")

    state = KillSwitch(path=kill_file, extra_probes={"flags": unreachable}).check()

    assert state.tripped is True
    assert state.source == "probe"
    assert "TimeoutError" in (state.reason or "")


def test_a_probe_can_trip_deliberately(kill_file: Path) -> None:
    switch = KillSwitch(path=kill_file, extra_probes={"budget": lambda: (True, "spend cap hit")})
    state = switch.check()
    assert state.tripped is True
    assert state.reason == "spend cap hit"


def test_probes_run_only_after_env_and_file(kill_file: Path) -> None:
    """Ordering matters: a cheap local check must not be gated behind a network call."""
    ran: list[str] = []

    def probe() -> tuple[bool, str | None]:
        ran.append("probe")
        return False, None

    kill_file.write_text("tripped\n")
    KillSwitch(path=kill_file, extra_probes={"p": probe}).check()
    assert ran == [], "the file sentinel already decided; the probe should not have run"


# ----------------------------------------------------------------- caching


def test_checks_are_uncached_by_default(kill_file: Path) -> None:
    """A stale *clear* reading is exactly the failure this component prevents."""
    switch = KillSwitch(path=kill_file)
    assert switch.check().tripped is False
    kill_file.write_text("now tripped\n")
    assert switch.check().tripped is True, "default must observe the change immediately"


def test_a_ttl_reuses_a_reading_until_it_expires(kill_file: Path) -> None:
    switch = KillSwitch(path=kill_file, cache_ttl=5.0)
    assert switch.check().tripped is False
    kill_file.write_text("now tripped\n")
    assert switch.check().tripped is False, "within the TTL, the cached reading stands"
    switch.invalidate()
    assert switch.check().tripped is True


def test_trip_and_clear_helpers_invalidate_the_cache(kill_file: Path) -> None:
    switch = KillSwitch(path=kill_file, cache_ttl=60.0)
    assert switch.check().tripped is False
    switch.trip("by an operator")
    assert switch.check().tripped is True
    switch.clear()
    assert switch.check().tripped is False


def test_clear_is_idempotent_when_there_is_no_sentinel(kill_file: Path) -> None:
    KillSwitch(path=kill_file).clear()  # must not raise


def test_trip_without_a_configured_path_refuses_rather_than_no_ops() -> None:
    with pytest.raises(RuntimeError, match="no kill-switch sentinel path"):
        KillSwitch().trip()


# ------------------------------------------------------- effect on the gate


def make_gate(sink: InMemoryAuditLog, kill_file: Path, calls: list) -> tuple[StageGate, dict]:
    gate = StageGate(
        audit=sink,
        kill_switch=KillSwitch(path=kill_file),
        approval=StaticApprovalHandler(True, actor="ops@example.com"),
    )

    @gate.capability("demo.act", stage=Stage.ACT, risk=RiskTier.HIGH, describe="act on {target}")
    def act(target: str) -> str:
        calls.append(target)
        return "done"

    @gate.capability("demo.suggest", stage=Stage.SUGGEST, describe="suggest on {target}")
    def suggest(target: str) -> str:
        calls.append(target)
        return "done"

    return gate, {"act": act, "suggest": suggest}


def test_a_tripped_switch_degrades_act_to_observe_without_raising(
    sink, kill_file: Path, calls
) -> None:
    _gate, caps = make_gate(sink, kill_file, calls)
    kill_file.write_text("incident 4412\n")

    result = caps["act"]("widget-1")

    assert calls == []
    assert result.outcome is Outcome.BLOCKED
    assert result.event.stage is Stage.OBSERVE
    assert result.event.degraded_from is Stage.ACT
    assert result.event.declared_stage is Stage.ACT


def test_a_blocked_call_still_records_its_full_intended_effect(
    sink, kill_file: Path, calls
) -> None:
    """The record of what it wanted to do is the reason degradation beats crashing."""
    _gate, caps = make_gate(sink, kill_file, calls)
    kill_file.write_text("incident 4412\n")

    caps["act"]("widget-1")

    event = sink.events[0]
    assert event.effect == "act on widget-1"
    assert event.arguments == {"target": "widget-1"}
    assert event.kill_switch == {
        "tripped": True,
        "source": "file",
        "reason": event.kill_switch["reason"],
    }
    assert "incident 4412" in event.kill_switch["reason"]


def test_the_switch_is_checked_before_suggest_too(sink, kill_file: Path, calls) -> None:
    """Do not wake a human to approve something that cannot run anyway."""
    gate, caps = make_gate(sink, kill_file, calls)
    handler = gate.approval
    kill_file.write_text("incident 4412\n")

    result = caps["suggest"]("widget-1")

    assert calls == []
    assert result.outcome is Outcome.BLOCKED
    assert handler.requests == [], "no human should have been asked"


def test_observe_calls_do_not_consult_the_switch_at_all(sink, kill_file: Path, calls) -> None:
    """Nothing can execute at OBSERVE, so there is nothing to guard."""
    gate = StageGate(audit=sink, kill_switch=KillSwitch(path=kill_file))

    @gate.capability("demo.shadow", stage=Stage.OBSERVE)
    def shadow() -> None:
        calls.append("ran")

    kill_file.write_text("incident 4412\n")
    result = shadow()

    assert result.outcome is Outcome.RECORDED
    assert result.event.kill_switch is None


def test_the_agent_keeps_running_after_a_trip(sink, kill_file: Path, calls) -> None:
    _gate, caps = make_gate(sink, kill_file, calls)
    assert caps["act"]("a").outcome is Outcome.EXECUTED
    kill_file.write_text("stop\n")
    assert caps["act"]("b").outcome is Outcome.BLOCKED
    kill_file.unlink()
    assert caps["act"]("c").outcome is Outcome.EXECUTED
    assert calls == ["a", "c"]
    assert len(sink.events) == 3


def test_a_switch_whose_check_raises_is_treated_as_tripped(sink, calls) -> None:
    class ExplodingSwitch:
        def check(self):
            raise OSError("cannot reach the control plane")

    gate = StageGate(audit=sink, kill_switch=ExplodingSwitch())

    @gate.capability("demo.act", stage=Stage.ACT)
    def act() -> None:
        calls.append("ran")

    result = act()

    assert calls == []
    assert result.outcome is Outcome.BLOCKED
    assert result.event.kill_switch["source"] == "error"


def test_env_var_name_is_configurable(kill_file: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MY_AGENT_STOP", "1")
    assert KillSwitch(path=kill_file, env_var="MY_AGENT_STOP").check().tripped is True
    assert KillSwitch(path=kill_file).check().tripped is False
