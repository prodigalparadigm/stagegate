"""Stage policy: resolution order, ceilings, and loading from configuration."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from stagegate import ConfigurationError, RiskTier, Stage, StagePolicy


def test_an_empty_policy_leaves_declared_stages_alone() -> None:
    resolution = StagePolicy().resolve("a.b", Stage.ACT, RiskTier.LOW)
    assert resolution.stage is Stage.ACT
    assert resolution.source == "declared"


def test_an_exact_override_beats_a_pattern() -> None:
    policy = StagePolicy(overrides={"jira.comment": Stage.ACT}, patterns={"jira.*": Stage.OBSERVE})
    assert policy.resolve("jira.comment", Stage.OBSERVE, RiskTier.LOW).source == "exact"
    assert policy.resolve("jira.other", Stage.ACT, RiskTier.LOW).stage is Stage.OBSERVE


def test_the_longest_matching_pattern_wins_regardless_of_ordering() -> None:
    policy = StagePolicy(patterns={"*": Stage.OBSERVE, "billing.*": Stage.SUGGEST})
    assert policy.resolve("billing.refund", Stage.ACT, RiskTier.LOW).stage is Stage.SUGGEST
    assert policy.resolve("other.thing", Stage.ACT, RiskTier.LOW).stage is Stage.OBSERVE


def test_patterns_are_case_sensitive() -> None:
    policy = StagePolicy(patterns={"Billing.*": Stage.ACT})
    assert policy.resolve("billing.refund", Stage.OBSERVE, RiskTier.LOW).source == "declared"


def test_a_ceiling_lowers_but_never_promotes() -> None:
    policy = StagePolicy(overrides={"a.b": Stage.ACT}, max_stage=Stage.SUGGEST)
    assert policy.resolve("a.b", Stage.OBSERVE, RiskTier.LOW).stage is Stage.SUGGEST

    policy = StagePolicy(max_stage=Stage.ACT)
    assert policy.resolve("a.b", Stage.OBSERVE, RiskTier.LOW).stage is Stage.OBSERVE


def test_a_risk_ceiling_applies_to_capabilities_that_do_not_exist_yet() -> None:
    policy = StagePolicy(max_stage_by_risk={RiskTier.CRITICAL: Stage.SUGGEST})
    assert policy.resolve("anything.new", Stage.ACT, RiskTier.CRITICAL).stage is Stage.SUGGEST
    assert policy.resolve("anything.new", Stage.ACT, RiskTier.HIGH).stage is Stage.ACT


def test_the_global_ceiling_is_applied_last_and_wins() -> None:
    policy = StagePolicy(
        overrides={"a.b": Stage.ACT},
        max_stage_by_risk={RiskTier.LOW: Stage.SUGGEST},
        max_stage=Stage.OBSERVE,
    )
    resolution = policy.resolve("a.b", Stage.ACT, RiskTier.LOW)
    assert resolution.stage is Stage.OBSERVE
    assert resolution.source == "max_stage"


def test_strings_are_accepted_everywhere_a_stage_is() -> None:
    policy = StagePolicy(
        overrides={"a.b": "act"}, patterns={"c.*": "suggest"}, max_stage_by_risk={"critical": "observe"}
    )
    assert policy.resolve("a.b", Stage.OBSERVE, RiskTier.LOW).stage is Stage.ACT
    assert policy.resolve("c.d", Stage.OBSERVE, RiskTier.CRITICAL).stage is Stage.OBSERVE


def test_a_bad_stage_in_configuration_fails_at_load_not_at_call_time() -> None:
    with pytest.raises(ConfigurationError, match="unknown stage"):
        StagePolicy(overrides={"a.b": "eventually"})


def test_toml_loading_including_a_nested_table(tmp_path: Path) -> None:
    path = tmp_path / "policy.toml"
    path.write_text(
        """
[stagegate]
max_stage = "suggest"
max_stage_by_risk = { critical = "observe" }

[stagegate.overrides]
"jira.comment" = "act"

[stagegate.patterns]
"jira.*" = "observe"
""",
        encoding="utf-8",
    )
    policy = StagePolicy.from_file(path)
    assert policy.max_stage is Stage.SUGGEST
    assert policy.resolve("jira.comment", Stage.OBSERVE, RiskTier.LOW).stage is Stage.SUGGEST
    assert policy.resolve("jira.other", Stage.ACT, RiskTier.LOW).stage is Stage.OBSERVE


def test_json_loading(tmp_path: Path) -> None:
    path = tmp_path / "policy.json"
    path.write_text(json.dumps({"overrides": {"a.b": "act"}}), encoding="utf-8")
    assert StagePolicy.from_file(path).resolve("a.b", Stage.OBSERVE, RiskTier.LOW).stage is Stage.ACT


def test_a_missing_policy_file_is_a_configuration_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="cannot read"):
        StagePolicy.from_file(tmp_path / "nope.toml")


def test_unparseable_policy_is_a_configuration_error(tmp_path: Path) -> None:
    path = tmp_path / "policy.toml"
    path.write_text("this is not = = toml", encoding="utf-8")
    with pytest.raises(ConfigurationError, match="cannot parse"):
        StagePolicy.from_file(path)


def test_a_typo_in_a_policy_key_is_caught_rather_than_ignored(tmp_path: Path) -> None:
    """Silently ignoring `overides` would leave a capability acting when nobody meant it to."""
    path = tmp_path / "policy.toml"
    path.write_text('[overides]\n"a.b" = "act"\n', encoding="utf-8")
    with pytest.raises(ConfigurationError, match="unknown stage policy keys: overides"):
        StagePolicy.from_file(path)


def test_describe_renders_the_policy_for_a_startup_log() -> None:
    policy = StagePolicy(overrides={"a.b": Stage.ACT}, max_stage=Stage.SUGGEST)
    described = policy.describe()
    assert "ceiling for all capabilities: suggest" in described
    assert "override a.b -> act" in described
    assert StagePolicy().describe() == [
        "no overrides; every capability runs at its declared stage"
    ]


def test_the_shipped_example_policy_loads() -> None:
    path = Path(__file__).resolve().parent.parent / "examples" / "policy.toml"
    policy = StagePolicy.from_file(path)
    assert policy.resolve("tickets.transition", Stage.ACT, RiskTier.HIGH).stage is Stage.OBSERVE
    assert policy.resolve("oncall.page", Stage.ACT, RiskTier.CRITICAL).stage is Stage.SUGGEST
    assert policy.resolve("tickets.search", Stage.OBSERVE, RiskTier.LOW).stage is Stage.ACT
