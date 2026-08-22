"""The dry-run report: "what would this agent have done?"

This is the artifact a change-advisory board reads. An agent runs in ``OBSERVE``
for two weeks against real traffic, and this turns the resulting audit log into
the document that says: here is every action it wanted to take, how often, with
what arguments, and here is what nobody has verified yet.

The report is deliberately not a recommendation. It assembles evidence and names
its own gaps; a human promotes the capability. The most important line in it is
the caveat that shadow mode proves *intent*, not *success* -- an ``OBSERVE`` call
never ran, so nothing here says the action would have worked. Reports that blur
that distinction are how organisations end up promoting a capability that
generates perfectly reasonable-looking requests against an endpoint that would
have rejected every one of them.
"""

from __future__ import annotations

import json
import math
import os
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .audit import AuditEvent, ChainVerification, read_events, utc_now, verify_chain
from .stages import Decision, Outcome, RiskTier, Stage

__all__ = ["DryRunReport", "CapabilityReport", "EffectGroup", "build_report", "report_from_log"]


def _percentile(values: Sequence[float], fraction: float) -> float | None:
    """Nearest-rank percentile: the smallest value at or above ``fraction`` of the sample.

    No interpolation. Samples here are small -- a capability with nine approvals
    has a p95, and inventing a value between two observed latencies would be
    presenting a number nobody measured to a change board.

    >>> _percentile([10.0, 20.0], 0.5), _percentile([10.0, 20.0], 0.95)
    (10.0, 20.0)
    >>> _percentile([], 0.5) is None
    True
    """
    if not values:
        return None
    ordered = sorted(values)
    rank = math.ceil(fraction * len(ordered))
    index = min(len(ordered) - 1, max(0, rank - 1))
    return round(ordered[index], 1)


@dataclass(frozen=True)
class EffectGroup:
    """A distinct intended effect and how often the agent wanted it.

    Attributes:
        effect: The rendered intended effect, as logged.
        count: How many calls produced this effect.
        example_arguments: Redacted arguments from the first such call.
        correlation_ids: Up to five runs that produced it, so a reviewer reading
            the report can grep the log back to the episodes it came from.
    """

    effect: str
    count: int
    example_arguments: Mapping[str, Any]
    correlation_ids: tuple[str, ...]


@dataclass
class CapabilityReport:
    """Everything the log says about one capability."""

    name: str
    risk_tier: RiskTier
    declared_stage: Stage
    stages_seen: Counter[str] = field(default_factory=Counter)
    outcomes: Counter[str] = field(default_factory=Counter)
    decisions: Counter[str] = field(default_factory=Counter)
    effects: list[EffectGroup] = field(default_factory=list)
    calls: int = 0
    runs: set[str] = field(default_factory=set)
    blocked: int = 0
    errors: list[Mapping[str, Any]] = field(default_factory=list)
    approval_latencies_ms: list[float] = field(default_factory=list)
    durations_ms: list[float] = field(default_factory=list)
    approvers: Counter[str] = field(default_factory=Counter)
    first_seen: str = ""
    last_seen: str = ""

    @property
    def shadow_calls(self) -> int:
        """Calls recorded at ``OBSERVE`` -- the evidence base for promotion."""
        return self.stages_seen.get(Stage.OBSERVE.value, 0)

    @property
    def distinct_effects(self) -> int:
        return len(self.effects)

    @property
    def approval_rate(self) -> float | None:
        """Share of approval requests that were approved, or ``None`` if none were made."""
        asked = sum(
            self.decisions.get(d.value, 0)
            for d in (Decision.APPROVED, Decision.DENIED, Decision.TIMED_OUT, Decision.ERROR)
        )
        if asked == 0:
            return None
        return round(self.decisions.get(Decision.APPROVED.value, 0) / asked, 3)

    def readiness(self) -> list[tuple[str, str, str]]:
        """Evidence for a promotion decision, as ``(check, status, detail)`` rows.

        Statuses are ``ok``, ``attention`` and ``unknown``. There is no ``pass``,
        and no overall verdict, because this function is not entitled to one.
        """
        rows: list[tuple[str, str, str]] = []

        if self.shadow_calls == 0:
            rows.append(("shadow volume", "attention", "no OBSERVE-stage calls recorded"))
        elif self.shadow_calls < 20:
            rows.append(
                ("shadow volume", "attention", f"{self.shadow_calls} shadow calls is a thin sample")
            )
        else:
            rows.append(("shadow volume", "ok", f"{self.shadow_calls} shadow calls"))

        if self.distinct_effects <= 1 and self.calls > 5:
            rows.append(
                ("effect variety", "attention",
                 "every call produced the same intended effect; the sample may not "
                 "exercise the interesting paths")
            )
        else:
            rows.append(
                ("effect variety", "ok", f"{self.distinct_effects} distinct intended effects")
            )

        rows.append(
            ("execution evidence", "unknown",
             "shadow calls never ran, so nothing here shows the action would have "
             "succeeded against the real system")
        )

        if self.blocked:
            rows.append(
                ("guard activations", "attention",
                 f"{self.blocked} call(s) blocked by a kill switch or unavailable audit sink")
            )

        if self.errors:
            rows.append(
                ("errors", "attention", f"{len(self.errors)} call(s) raised during execution")
            )

        if self.risk_tier.rank >= RiskTier.HIGH.rank:
            rows.append(
                ("risk tier", "attention",
                 f"declared {self.risk_tier.value}; promotion past SUGGEST should be an "
                 "explicit, named decision")
            )
        return rows


@dataclass
class DryRunReport:
    """Aggregate view of an audit log, renderable as Markdown or JSON."""

    generated_at: str
    source: str
    total_events: int
    capabilities: list[CapabilityReport]
    outcomes: Counter[str]
    stages: Counter[str]
    decisions: Counter[str]
    runs: set[str]
    kill_switch_events: list[Mapping[str, Any]]
    window: tuple[str, str] | None
    verification: ChainVerification | None = None
    actors: Counter[str] = field(default_factory=Counter)

    # ------------------------------------------------------------------ views

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable form, for feeding a dashboard or a diff."""
        return {
            "schema": "stagegate.report/1",
            "generated_at": self.generated_at,
            "source": self.source,
            "window": {"first_event": self.window[0], "last_event": self.window[1]}
            if self.window
            else None,
            "total_events": self.total_events,
            "total_runs": len(self.runs),
            "integrity": {
                "verified": self.verification.ok if self.verification else None,
                "records": self.verification.count if self.verification else None,
                "reason": self.verification.reason if self.verification else None,
                "head_hash": self.verification.head_hash if self.verification else None,
            },
            "outcomes": dict(self.outcomes),
            "stages": dict(self.stages),
            "decisions": dict(self.decisions),
            "actors": dict(self.actors),
            "kill_switch_events": list(self.kill_switch_events),
            "capabilities": [
                {
                    "capability": cap.name,
                    "risk_tier": cap.risk_tier.value,
                    "declared_stage": cap.declared_stage.value,
                    "calls": cap.calls,
                    "runs": len(cap.runs),
                    "shadow_calls": cap.shadow_calls,
                    "stages_seen": dict(cap.stages_seen),
                    "outcomes": dict(cap.outcomes),
                    "decisions": dict(cap.decisions),
                    "approval_rate": cap.approval_rate,
                    "approval_latency_p50_ms": _percentile(cap.approval_latencies_ms, 0.5),
                    "approval_latency_p95_ms": _percentile(cap.approval_latencies_ms, 0.95),
                    "duration_p50_ms": _percentile(cap.durations_ms, 0.5),
                    "duration_p95_ms": _percentile(cap.durations_ms, 0.95),
                    "blocked": cap.blocked,
                    "errors": list(cap.errors),
                    "approvers": dict(cap.approvers),
                    "first_seen": cap.first_seen,
                    "last_seen": cap.last_seen,
                    "intended_effects": [
                        {
                            "effect": group.effect,
                            "count": group.count,
                            "example_arguments": dict(group.example_arguments),
                            "example_runs": list(group.correlation_ids),
                        }
                        for group in cap.effects
                    ],
                    "readiness": [
                        {"check": check, "status": status, "detail": detail}
                        for check, status, detail in cap.readiness()
                    ],
                }
                for cap in self.capabilities
            ],
        }

    def to_json(self, *, indent: int = 2) -> str:
        """Render as JSON."""
        return json.dumps(self.to_dict(), indent=indent, sort_keys=False, default=str)

    def to_markdown(self, *, max_effects: int = 10) -> str:
        """Render the document a human actually reads before signing off."""
        out: list[str] = []
        add = out.append

        add("# Agent dry-run report")
        add("")
        add(f"- Source: `{self.source}`")
        add(f"- Generated: {self.generated_at}")
        if self.window:
            add(f"- Window: {self.window[0]} to {self.window[1]}")
        add(f"- Events: {self.total_events} across {len(self.runs)} agent run(s)")
        if self.verification is not None:
            if self.verification.ok:
                add(
                    f"- Integrity: **verified**, {self.verification.count} records chained "
                    f"(head `{(self.verification.head_hash or '')[:16]}...`)"
                )
            else:
                add(
                    f"- Integrity: **FAILED at record {self.verification.first_bad_seq}** - "
                    f"{self.verification.reason}"
                )
        add("")
        add(
            "> Shadow-mode records show what the agent *intended*. Nothing in an "
            "`observe` record demonstrates that the action would have succeeded "
            "against the real system, because it never ran."
        )
        add("")

        add("## Summary")
        add("")
        add("| Outcome | Count |")
        add("| --- | --- |")
        for name, count in self.outcomes.most_common():
            add(f"| `{name}` | {count} |")
        add("")

        if self.decisions:
            add("| Human decision | Count |")
            add("| --- | --- |")
            for name, count in self.decisions.most_common():
                add(f"| `{name}` | {count} |")
            add("")

        add("## Capabilities")
        add("")
        add(
            "| Capability | Risk | Declared | Calls | Shadow | "
            "Distinct effects | Blocked | Errors |"
        )
        add("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |")
        for cap in self.capabilities:
            add(
                f"| `{cap.name}` | {cap.risk_tier.value} | {cap.declared_stage.value} | "
                f"{cap.calls} | {cap.shadow_calls} | {cap.distinct_effects} | "
                f"{cap.blocked} | {len(cap.errors)} |"
            )
        add("")

        if self.kill_switch_events:
            add("## Kill-switch activations")
            add("")
            for entry in self.kill_switch_events[:20]:
                add(
                    f"- `{entry.get('timestamp')}` `{entry.get('capability')}` "
                    f"(source: {entry.get('source')}) - {entry.get('reason')}"
                )
            if len(self.kill_switch_events) > 20:
                add(f"- ...and {len(self.kill_switch_events) - 20} more")
            add("")

        add("## What the agent would have done")
        add("")
        for cap in self.capabilities:
            add(f"### `{cap.name}`")
            add("")
            add(
                f"Risk `{cap.risk_tier.value}` - declared stage `{cap.declared_stage.value}` - "
                f"{cap.calls} call(s) across {len(cap.runs)} run(s)"
            )
            add("")
            if cap.effects:
                add("| Intended effect | Times |")
                add("| --- | ---: |")
                for group in cap.effects[:max_effects]:
                    add(f"| {_md_cell(group.effect)} | {group.count} |")
                if len(cap.effects) > max_effects:
                    add(f"| _...{len(cap.effects) - max_effects} more distinct effects_ | |")
                add("")
                first = cap.effects[0]
                add("Example arguments (redacted as logged):")
                add("")
                add("```json")
                add(json.dumps(dict(first.example_arguments), indent=2, default=str))
                add("```")
                add("")
            else:
                add("_No intended effects recorded._")
                add("")

            if cap.approval_latencies_ms:
                add(
                    f"Approval latency: p50 {_percentile(cap.approval_latencies_ms, 0.5)} ms, "
                    f"p95 {_percentile(cap.approval_latencies_ms, 0.95)} ms"
                    + (
                        f" - approval rate {cap.approval_rate:.0%}"
                        if cap.approval_rate is not None
                        else ""
                    )
                )
                if cap.approvers:
                    who = ", ".join(f"{name} ({n})" for name, n in cap.approvers.most_common(5))
                    add(f"Approvers: {who}")
                add("")

            if cap.errors:
                add("Errors during execution:")
                add("")
                for error in cap.errors[:5]:
                    add(f"- `{error.get('type')}`: {_md_cell(str(error.get('message', '')))}")
                add("")

            add("Promotion evidence:")
            add("")
            add("| Check | Status | Detail |")
            add("| --- | --- | --- |")
            for check, status, detail in cap.readiness():
                add(f"| {check} | `{status}` | {_md_cell(detail)} |")
            add("")

        add("## Sign-off")
        add("")
        add(
            "This report is evidence, not a recommendation. Promotion of a capability "
            "from `observe` to `suggest` or `act` is a configuration change that a named "
            "person approves. Record who, when, and against which report head hash:"
        )
        add("")
        head = (self.verification.head_hash if self.verification else None) or "(not verified)"
        add(f"- Report head hash: `{head}`")
        add("- Capability promoted: ")
        add("- From stage / to stage: ")
        add("- Approved by / date: ")
        add("")
        return "\n".join(out)


def _md_cell(text: str) -> str:
    """Make a string safe to drop into a Markdown table cell."""
    cleaned = str(text).replace("|", "\\|").replace("\n", " ").strip()
    return cleaned if len(cleaned) <= 300 else cleaned[:297] + "..."


def build_report(
    events: Iterable[AuditEvent],
    *,
    source: str = "(in memory)",
    verification: ChainVerification | None = None,
    max_effect_groups: int = 200,
) -> DryRunReport:
    """Aggregate events into a :class:`DryRunReport`.

    Args:
        events: Audit events, in log order.
        source: Where they came from, for the report header.
        verification: Chain verification result to include, if one was run.
        max_effect_groups: Cap on distinct effects tracked per capability, so a
            capability whose effect string embeds a unique id cannot make the
            report unbounded. Overflow is counted under a single bucket rather
            than dropped, so the totals still add up.
    """
    by_name: dict[str, CapabilityReport] = {}
    effect_counts: dict[str, Counter[str]] = defaultdict(Counter)
    effect_examples: dict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    effect_runs: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    outcomes: Counter[str] = Counter()
    stages: Counter[str] = Counter()
    decisions: Counter[str] = Counter()
    actors: Counter[str] = Counter()
    runs: set[str] = set()
    kill_switch_events: list[Mapping[str, Any]] = []
    timestamps: list[str] = []
    total = 0

    for event in events:
        total += 1
        outcomes[event.outcome.value] += 1
        stages[event.stage.value] += 1
        if event.decision is not Decision.NOT_REQUIRED:
            decisions[event.decision.value] += 1
        if event.correlation_id:
            runs.add(event.correlation_id)
        if event.actor:
            actors[event.actor] += 1
        if event.timestamp:
            timestamps.append(event.timestamp)

        cap = by_name.get(event.capability)
        if cap is None:
            cap = CapabilityReport(
                name=event.capability,
                risk_tier=event.risk_tier,
                declared_stage=event.declared_stage,
                first_seen=event.timestamp,
            )
            by_name[event.capability] = cap
        cap.calls += 1
        cap.last_seen = event.timestamp or cap.last_seen
        cap.stages_seen[event.stage.value] += 1
        cap.outcomes[event.outcome.value] += 1
        if event.correlation_id:
            cap.runs.add(event.correlation_id)
        if event.decision is not Decision.NOT_REQUIRED:
            cap.decisions[event.decision.value] += 1
        if event.decision_actor:
            cap.approvers[event.decision_actor] += 1
        if event.approval_latency_ms is not None:
            cap.approval_latencies_ms.append(float(event.approval_latency_ms))
        if event.duration_ms is not None:
            cap.durations_ms.append(float(event.duration_ms))
        if event.outcome is Outcome.BLOCKED:
            cap.blocked += 1
        if event.error:
            cap.errors.append(event.error)

        if event.kill_switch and event.kill_switch.get("tripped"):
            kill_switch_events.append(
                {
                    "timestamp": event.timestamp,
                    "capability": event.capability,
                    "correlation_id": event.correlation_id,
                    "source": event.kill_switch.get("source"),
                    "reason": event.kill_switch.get("reason"),
                }
            )

        effect = event.effect or f"{event.capability}(...)"
        bucket = effect_counts[event.capability]
        if effect not in bucket and len(bucket) >= max_effect_groups:
            effect = f"[{max_effect_groups}+ distinct effects; further variants grouped]"
        bucket[effect] += 1
        effect_examples[event.capability].setdefault(effect, event.arguments)
        if event.correlation_id and len(effect_runs[event.capability][effect]) < 5:
            effect_runs[event.capability][effect].append(event.correlation_id)

    for name, cap in by_name.items():
        cap.effects = [
            EffectGroup(
                effect=effect,
                count=count,
                example_arguments=effect_examples[name].get(effect, {}),
                correlation_ids=tuple(effect_runs[name].get(effect, ())),
            )
            for effect, count in effect_counts[name].most_common()
        ]

    window = (min(timestamps), max(timestamps)) if timestamps else None
    ordered = sorted(
        by_name.values(), key=lambda c: (-c.risk_tier.rank, -c.calls, c.name)
    )
    return DryRunReport(
        generated_at=utc_now(),
        source=source,
        total_events=total,
        capabilities=ordered,
        outcomes=outcomes,
        stages=stages,
        decisions=decisions,
        runs=runs,
        kill_switch_events=kill_switch_events,
        window=window,
        verification=verification,
        actors=actors,
    )


def report_from_log(path: str | os.PathLike[str], *, verify: bool = True) -> DryRunReport:
    """Build a report straight from a JSONL audit log.

    Args:
        path: The log to read.
        verify: Also check the hash chain and put the result in the header. On by
            default: a report from a log nobody checked is worth very little, and
            the check costs one extra pass.
    """
    verification = verify_chain(path) if verify else None
    return build_report(read_events(path), source=str(path), verification=verification)
