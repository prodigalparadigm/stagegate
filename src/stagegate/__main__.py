"""Command line entry point: ``stagegate report`` and ``stagegate verify``.

Both read a JSONL audit log and neither needs the agent that produced it, which
is the point -- an auditor gets the log, not your codebase.

Exit codes: ``0`` success, ``1`` verification failed, ``2`` usage or I/O error.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from .audit import verify_chain
from .report import report_from_log

__all__ = ["main"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="stagegate",
        description="Inspect a StageGate audit log.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    report = sub.add_parser(
        "report",
        help="build the 'what would this agent have done' dry-run report",
    )
    report.add_argument("log", help="path to a JSONL audit log")
    report.add_argument(
        "--format", choices=("markdown", "json"), default="markdown", help="output format"
    )
    report.add_argument("-o", "--output", help="write here instead of stdout")
    report.add_argument(
        "--no-verify", action="store_true", help="skip the hash-chain check (not advised)"
    )
    report.add_argument(
        "--fail-on-tamper",
        action="store_true",
        help="exit 1 if the chain does not verify (use in CI)",
    )

    verify = sub.add_parser("verify", help="check an audit log's hash chain")
    verify.add_argument("log", help="path to a JSONL audit log")
    verify.add_argument(
        "--quiet", action="store_true", help="print nothing; rely on the exit code"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI. Returns the process exit code."""
    args = build_parser().parse_args(argv)

    if args.command == "verify":
        try:
            result = verify_chain(args.log)
        except OSError as exc:
            # verify_chain reports a *missing* log as a result; anything else
            # (a directory, a permission error, a bad mount) is an I/O problem
            # with the invocation, not a verdict on the chain.
            print(f"cannot read {args.log}: {exc}", file=sys.stderr)
            return 2
        if not args.quiet:
            if result.ok:
                plural = "" if result.count == 1 else "s"
                print(f"OK: {result.count} record{plural} verified; head {result.head_hash}")
            else:
                print(
                    f"FAILED at record {result.first_bad_seq}: {result.reason}",
                    file=sys.stderr,
                )
        return 0 if result.ok else 1

    try:
        report = report_from_log(args.log, verify=not args.no_verify)
    except OSError as exc:
        print(f"cannot read {args.log}: {exc}", file=sys.stderr)
        return 2

    text = report.to_json() if args.format == "json" else report.to_markdown()
    if args.output:
        try:
            with open(args.output, "w", encoding="utf-8") as handle:
                handle.write(text + "\n")
        except OSError as exc:
            print(f"cannot write {args.output}: {exc}", file=sys.stderr)
            return 2
        print(f"wrote {args.output}", file=sys.stderr)
    else:
        print(text)

    if args.fail_on_tamper and report.verification is not None and not report.verification.ok:
        print(
            f"integrity check failed at record {report.verification.first_bad_seq}: "
            f"{report.verification.reason}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
