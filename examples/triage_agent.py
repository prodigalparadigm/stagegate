"""A small triage agent whose capabilities sit at three different stages.

Run it:

    python examples/triage_agent.py                # scripted approvals
    python examples/triage_agent.py --interactive  # approve at the terminal
    python examples/triage_agent.py --tripped      # with the kill switch on

No network, no credentials, no external services. The "backends" are dictionaries
in this file. Everything else -- the gate, the policy, the audit chain, the
approval flow, the report -- is the real library.

What to look at, in order:

1. Four capabilities, declared at different stages with different risk tiers.
2. ``examples/policy.toml``, which is where the effective stages actually come
   from. Note that ``tickets.transition`` is declared ``ACT`` in code and still
   runs at ``OBSERVE``, because the policy says so and code does not get a vote.
3. The audit log printed at the end, and the dry-run report built from it.
4. ``triage.escalate`` calling ``tickets.comment``: nested calls share one
   correlation id and the inner event points at the outer one.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from stagegate import (  # noqa: E402
    CLIApprovalHandler,
    JsonlAuditLog,
    KillSwitch,
    RedactionPolicy,
    RiskTier,
    Stage,
    StageGate,
    StagePolicy,
    StaticApprovalHandler,
    agent_run,
    report_from_log,
    verify_chain,
)

# --------------------------------------------------------------------------
# Fake backends. In a real deployment these are the client's ticketing system,
# their paging provider, their CRM. Nothing about the gating changes.
# --------------------------------------------------------------------------

TICKETS: dict[str, dict[str, Any]] = {
    "T-1001": {"title": "Checkout returns 500 for EU cards", "status": "open", "priority": "P1"},
    "T-1002": {"title": "Docs typo on the pricing page", "status": "open", "priority": "P4"},
    "T-1003": {"title": "Export job silently truncates at 10k rows", "status": "open", "priority": "P2"},
}
COMMENTS: list[tuple[str, str]] = []
PAGES: list[str] = []


def build_gate(log_path: Path, *, interactive: bool, kill_file: Path) -> StageGate:
    """Wire up the gate. This is the whole configuration surface."""
    approval = (
        CLIApprovalHandler(actor="operator@example.com")
        if interactive
        else StaticApprovalHandler(True, actor="scripted@example.com", note="demo auto-approval")
    )
    return StageGate(
        audit=JsonlAuditLog(log_path, fsync=False),
        policy=StagePolicy.from_file(Path(__file__).with_name("policy.toml")),
        kill_switch=KillSwitch(path=kill_file),
        approval=approval,
        approval_timeout=60.0 if interactive else 5.0,
        redaction=RedactionPolicy(),
        labels={"deployment": "demo", "service": "triage-agent"},
    )


def register_capabilities(gate: StageGate) -> dict[str, Any]:
    """Declare what the agent can do, and how far each thing is trusted."""

    @gate.capability(
        "tickets.search",
        stage=Stage.OBSERVE,
        risk=RiskTier.LOW,
        describe="Search tickets matching {query!r}",
    )
    def search_tickets(query: str, api_token: str = "sk-demo-000000000000000000") -> list[str]:
        """Find open tickets. Read-only; promoted to ACT by policy."""
        query = query.lower()
        return [
            key for key, value in TICKETS.items()
            if query in value["title"].lower() or query in value["priority"].lower()
        ]

    @gate.capability(
        "tickets.comment",
        stage=Stage.OBSERVE,
        risk=RiskTier.MODERATE,
        describe="Comment on {ticket_id}: {body}",
    )
    def comment_on_ticket(ticket_id: str, body: str) -> dict[str, Any]:
        """Add a public comment. Visible to the reporter, so a human signs off."""
        if ticket_id not in TICKETS:
            raise KeyError(f"no such ticket: {ticket_id}")
        COMMENTS.append((ticket_id, body))
        return {"ticket_id": ticket_id, "comment_index": len(COMMENTS)}

    @gate.capability(
        "tickets.transition",
        stage=Stage.ACT,
        risk=RiskTier.HIGH,
        describe="Move {ticket_id} from open to {status}",
    )
    def transition_ticket(ticket_id: str, status: str) -> dict[str, Any]:
        """Change a ticket's state. Declared ACT in code; policy holds it at OBSERVE."""
        TICKETS[ticket_id]["status"] = status
        return {"ticket_id": ticket_id, "status": status}

    @gate.capability(
        "oncall.page",
        stage=Stage.ACT,
        risk=RiskTier.CRITICAL,
        describe="Page {rotation} about {ticket_id}",
    )
    def page_oncall(rotation: str, ticket_id: str, reporter_email: str = "") -> str:
        """Wake somebody up. Capped at SUGGEST by the risk ceiling in policy.toml."""
        PAGES.append(f"{rotation}:{ticket_id}")
        return f"paged {rotation}"

    @gate.capability(
        "triage.escalate",
        stage=Stage.ACT,
        risk=RiskTier.MODERATE,
        describe="Escalate {ticket_id}",
    )
    def escalate(ticket_id: str) -> str:
        """Composite action: comments, then transitions. Demonstrates nesting."""
        comment_on_ticket(ticket_id, "Escalating: matches a known P1 pattern.")
        transition_ticket(ticket_id, "escalated")
        return ticket_id

    return {
        "search": search_tickets,
        "comment": comment_on_ticket,
        "transition": transition_ticket,
        "page": page_oncall,
        "escalate": escalate,
    }


def run_agent(gate: StageGate, caps: dict[str, Any]) -> None:
    """One agent run. Everything below shares a single correlation id."""
    with agent_run(actor="triage-agent@example.com", labels={"ticket_batch": "morning"}) as run:
        print(f"\n--- agent run {run.correlation_id} ---\n")

        found = caps["search"]("P1")
        print(f"tickets.search   -> {found.outcome.value:9} value={found.value_or([])}")

        for ticket_id in found.value_or([]):
            commented = caps["comment"](ticket_id, "Triage: reproduced on staging.")
            print(f"tickets.comment  -> {commented.outcome.value:9} decision={commented.event.decision.value}")

            moved = caps["transition"](ticket_id, "in_progress")
            print(
                f"tickets.transition -> {moved.outcome.value:9} "
                f"(declared {moved.event.declared_stage.value}, ran at {moved.event.stage.value}) "
                f"| intended: {moved.event.effect}"
            )

            paged = caps["page"]("platform-primary", ticket_id, reporter_email="ada@example.com")
            print(f"oncall.page      -> {paged.outcome.value:9} decision={paged.event.decision.value}")

        escalated = caps["escalate"]("T-1003")
        print(f"triage.escalate  -> {escalated.outcome.value:9} (nested calls share the run id)")

        # A capability invoked by name, the way a tool-calling model dispatches.
        by_name = gate.invoke("tickets.search", "export")
        print(f"invoke('tickets.search') -> {by_name.outcome.value:9} value={by_name.value_or([])}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--interactive", action="store_true", help="approve at the terminal")
    parser.add_argument("--tripped", action="store_true", help="start with the kill switch tripped")
    parser.add_argument("--report", action="store_true", help="print the full dry-run report")
    args = parser.parse_args(argv)

    workdir = Path(tempfile.mkdtemp(prefix="stagegate-demo-"))
    log_path = workdir / "audit.jsonl"
    kill_file = workdir / "kill"
    if args.tripped:
        kill_file.write_text("demo: incident 4412, agent actions suspended\n", encoding="utf-8")

    gate = build_gate(log_path, interactive=args.interactive, kill_file=kill_file)
    caps = register_capabilities(gate)

    print("Capability manifest (declared stage vs. what policy resolves to now):\n")
    print(f"  {'capability':<20} {'declared':<9} {'effective':<10} {'source':<22} risk")
    for row in gate.manifest():
        print(
            f"  {row['capability']:<20} {row['declared_stage']:<9} {row['effective_stage']:<10} "
            f"{row['stage_source']:<22} {row['risk_tier']}"
        )

    run_agent(gate, caps)
    gate.audit.close()

    print("\n--- side effects that actually happened ---")
    print(f"comments: {COMMENTS}")
    print(f"pages   : {PAGES}")
    print(f"statuses: { {k: v['status'] for k, v in TICKETS.items()} }")

    check = verify_chain(log_path)
    print(f"\naudit chain: ok={check.ok} records={check.count} head={(check.head_hash or '')[:16]}...")
    print(f"audit log  : {log_path}")

    report = report_from_log(log_path)
    if args.report:
        print("\n" + report.to_markdown())
    else:
        print("\nDry-run report (re-run with --report for the whole thing):\n")
        for cap in report.capabilities:
            effects = ", ".join(f"{g.effect} (x{g.count})" for g in cap.effects[:2])
            print(f"  {cap.name:<20} calls={cap.calls} shadow={cap.shadow_calls}  {effects}")
        print(f"\n  stagegate report {log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
