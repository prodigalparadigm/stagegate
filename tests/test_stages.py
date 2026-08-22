"""The stage vocabulary: ordering, parsing, and wire-format stability."""

from __future__ import annotations

import pytest

from stagegate import Decision, Outcome, RiskTier, Stage


def test_stages_are_ordered_by_promotion() -> None:
    assert Stage.OBSERVE < Stage.SUGGEST < Stage.ACT
    assert Stage.ACT > Stage.OBSERVE
    assert Stage.SUGGEST >= Stage.SUGGEST
    assert max(Stage.OBSERVE, Stage.ACT, key=lambda s: s.rank) is Stage.ACT


def test_stage_comparison_with_a_non_stage_is_not_implemented() -> None:
    with pytest.raises(TypeError):
        _ = Stage.ACT < "act"  # type: ignore[operator]


@pytest.mark.parametrize("raw", ["ACT", " act ", "Act", Stage.ACT])
def test_stage_parse_is_forgiving_about_case_and_whitespace(raw: str | Stage) -> None:
    assert Stage.parse(raw) is Stage.ACT


def test_stage_parse_rejects_nonsense_and_says_what_is_valid() -> None:
    with pytest.raises(ValueError, match="observe, suggest, act"):
        Stage.parse("yolo")


def test_risk_tiers_are_ordered() -> None:
    assert RiskTier.LOW.rank < RiskTier.MODERATE.rank < RiskTier.HIGH.rank < RiskTier.CRITICAL.rank
    assert RiskTier.parse("CRITICAL") is RiskTier.CRITICAL


def test_risk_parse_rejects_nonsense() -> None:
    with pytest.raises(ValueError, match="low, moderate, high, critical"):
        RiskTier.parse("spicy")


def test_only_execution_outcomes_report_as_executed() -> None:
    assert Outcome.EXECUTED.executed
    assert Outcome.FAILED.executed
    for outcome in (
        Outcome.RECORDED, Outcome.DENIED, Outcome.TIMED_OUT, Outcome.BLOCKED, Outcome.ERROR,
    ):
        assert not outcome.executed


def test_enum_values_are_the_documented_wire_format() -> None:
    """These strings appear in stored audit logs. Renaming one breaks every reader."""
    assert [s.value for s in Stage] == ["observe", "suggest", "act"]
    assert [r.value for r in RiskTier] == ["low", "moderate", "high", "critical"]
    assert [d.value for d in Decision] == [
        "not_required", "approved", "denied", "timed_out", "error",
    ]
    assert [o.value for o in Outcome] == [
        "recorded", "executed", "failed", "denied", "timed_out", "blocked", "error",
    ]
