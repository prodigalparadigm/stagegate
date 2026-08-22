"""Stage policy: where a capability's *effective* stage actually comes from.

The stage in the decorator is a default, not a decision. Promotion has to be
explicit configuration a reviewer can read, diff and approve, which means it has
to live outside the code that gets deployed with the agent.

Resolution order, most specific first:

1. An exact-name override (``"jira.transition_issue"``).
2. The longest matching glob pattern (``"jira.*"`` beats ``"*"``).
3. The stage declared on the decorator.

Then two ceilings are applied, and both can only lower the result:

* ``max_stage`` -- an environment-wide cap. Set it to ``OBSERVE`` in staging and
  no capability in that deployment can act, whatever the code or the overrides
  say.
* ``max_stage_by_risk`` -- a per-tier cap. "Nothing ``critical`` goes past
  ``SUGGEST``" is one line of configuration, and it holds for capabilities nobody
  has written yet.

A ceiling never promotes. A policy can always make a capability *safer* than its
declaration; it takes an explicit override to make it more dangerous, and that
override is a line in a file someone signed off on.
"""

from __future__ import annotations

import fnmatch
import json
import os
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .errors import ConfigurationError
from .stages import RiskTier, Stage

__all__ = ["StagePolicy", "StageResolution"]


@dataclass(frozen=True, slots=True)
class StageResolution:
    """How an effective stage was arrived at, so the reasoning is auditable.

    Attributes:
        stage: The stage in force.
        declared: The stage on the decorator.
        source: What decided: ``"declared"``, ``"exact"``, ``"pattern:<glob>"``,
            ``"max_stage"``, or ``"max_stage_by_risk"``.
    """

    stage: Stage
    declared: Stage
    source: str


@dataclass(frozen=True)
class StagePolicy:
    """Resolves declared stages into effective ones.

    Args:
        overrides: Exact capability name to stage.
        patterns: Glob pattern to stage. Longest pattern wins, so a specific rule
            beats a broad one regardless of dict ordering.
        max_stage: Ceiling for every capability.
        max_stage_by_risk: Ceiling per risk tier.

    Raises:
        ConfigurationError: on an unparseable stage or risk tier.

    Example:
        >>> policy = StagePolicy(
        ...     overrides={"jira.comment": Stage.ACT},
        ...     patterns={"billing.*": Stage.SUGGEST},
        ...     max_stage_by_risk={RiskTier.CRITICAL: Stage.SUGGEST},
        ... )
        >>> policy.resolve("jira.comment", Stage.OBSERVE, RiskTier.LOW).stage
        <Stage.ACT: 'act'>
    """

    overrides: Mapping[str, Stage] = field(default_factory=dict)
    patterns: Mapping[str, Stage] = field(default_factory=dict)
    max_stage: Stage | None = None
    max_stage_by_risk: Mapping[RiskTier, Stage] = field(default_factory=dict)

    def __post_init__(self) -> None:
        try:
            object.__setattr__(
                self, "overrides", {str(k): Stage.parse(v) for k, v in self.overrides.items()}
            )
            object.__setattr__(
                self, "patterns", {str(k): Stage.parse(v) for k, v in self.patterns.items()}
            )
            object.__setattr__(
                self,
                "max_stage_by_risk",
                {RiskTier.parse(k): Stage.parse(v) for k, v in self.max_stage_by_risk.items()},
            )
            if self.max_stage is not None:
                object.__setattr__(self, "max_stage", Stage.parse(self.max_stage))
        except ValueError as exc:
            raise ConfigurationError(str(exc)) from exc

    def resolve(self, name: str, declared: Stage, risk: RiskTier) -> StageResolution:
        """Return the effective stage for ``name`` and how it was decided."""
        stage = declared
        source = "declared"

        if name in self.overrides:
            stage = self.overrides[name]
            source = "exact"
        else:
            best: tuple[int, str, Stage] | None = None
            for pattern, candidate in self.patterns.items():
                if fnmatch.fnmatchcase(name, pattern):
                    score = len(pattern)
                    if best is None or score > best[0]:
                        best = (score, pattern, candidate)
            if best is not None:
                stage = best[2]
                source = f"pattern:{best[1]}"

        risk_ceiling = self.max_stage_by_risk.get(risk)
        if risk_ceiling is not None and stage > risk_ceiling:
            stage, source = risk_ceiling, "max_stage_by_risk"

        if self.max_stage is not None and stage > self.max_stage:
            stage, source = self.max_stage, "max_stage"

        return StageResolution(stage=stage, declared=declared, source=source)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> StagePolicy:
        """Build a policy from a plain mapping (parsed TOML, JSON, or a dict)."""
        unknown = set(data) - {"overrides", "patterns", "max_stage", "max_stage_by_risk"}
        if unknown:
            raise ConfigurationError(
                f"unknown stage policy keys: {', '.join(sorted(unknown))}. "
                "Expected: overrides, patterns, max_stage, max_stage_by_risk."
            )
        return cls(
            overrides=data.get("overrides") or {},
            patterns=data.get("patterns") or {},
            max_stage=data.get("max_stage"),
            max_stage_by_risk=data.get("max_stage_by_risk") or {},
        )

    @classmethod
    def from_file(cls, path: str | os.PathLike[str]) -> StagePolicy:
        """Load a policy from a ``.toml`` or ``.json`` file.

        The file may nest the policy under a ``[stagegate]`` table, so a policy can
        live inside an existing project config instead of a file of its own.

        Raises:
            ConfigurationError: if the file is missing, unparseable, or malformed.
        """
        file_path = Path(path)
        try:
            raw = file_path.read_bytes()
        except OSError as exc:
            raise ConfigurationError(f"cannot read stage policy {file_path}: {exc}") from exc

        try:
            if file_path.suffix == ".json":
                data = json.loads(raw.decode("utf-8"))
            else:
                data = tomllib.loads(raw.decode("utf-8"))
        except (ValueError, UnicodeDecodeError) as exc:
            raise ConfigurationError(f"cannot parse stage policy {file_path}: {exc}") from exc

        if not isinstance(data, dict):
            raise ConfigurationError(f"stage policy {file_path} must be a table/object")
        if "stagegate" in data and isinstance(data["stagegate"], dict):
            data = data["stagegate"]
        return cls.from_mapping(data)

    def describe(self) -> list[str]:
        """Render the policy as readable lines, for a report or a startup log."""
        lines: list[str] = []
        if self.max_stage is not None:
            lines.append(f"ceiling for all capabilities: {self.max_stage.value}")
        for risk, stage in sorted(self.max_stage_by_risk.items(), key=lambda kv: kv[0].rank):
            lines.append(f"ceiling for risk={risk.value}: {stage.value}")
        for name, stage in sorted(self.overrides.items()):
            lines.append(f"override {name} -> {stage.value}")
        for pattern, stage in sorted(self.patterns.items()):
            lines.append(f"pattern  {pattern} -> {stage.value}")
        return lines or ["no overrides; every capability runs at its declared stage"]
