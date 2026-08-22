"""Kill switch: two independent ways to stop an agent acting, both fail-closed.

Whoever is on call at 3am is not going to redeploy your agent. They need a way to
stop it that works from a shell, and a way that works from configuration
management, and they need to know which one wins. This module is those two ways.

**Precedence.** Evaluated in this order; the first source that trips, wins:

1. ``STAGEGATE_KILL`` -- the environment variable. A recognised true value trips.
   An *unrecognised* value also trips, because a switch that silently ignores
   ``STAGEGATE_KILL=yes-please`` is not a switch. A recognised false value
   *abstains*: it moves on to the file sentinel rather than clearing it.
2. The file sentinel. If the file exists, the switch is tripped and the file's
   first line is recorded as the reason.
3. If neither trips, the switch is clear.

The asymmetry in step 1 is deliberate. Anyone with a shell can trip the switch,
which is what you want in an incident; nobody with a shell can *clear* a sentinel
that configuration management or an incident commander put there. Stopping is
cheap and reversible, so make it easy. Resuming is a decision, so make it
deliberate.

**Fail-closed.** Anything the switch cannot determine counts as tripped: an
unparseable env value, a sentinel path that raises something other than "does not
exist", a custom probe that throws. The cost of a false trip is that an agent
degrades to shadow mode and someone gets paged. The cost of a false clear is an
agent acting on production during an incident.

**Degradation, not exceptions.** A tripped switch never raises into the agent. It
lowers the effective stage to ``OBSERVE``, so the call is still recorded with its
full intended effect and the agent keeps running. When the incident is over you
have a complete record of what it wanted to do while it was stopped.
"""

from __future__ import annotations

import os
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

__all__ = ["KillSwitchState", "KillSwitch", "TRUE_VALUES", "FALSE_VALUES"]

TRUE_VALUES: frozenset[str] = frozenset({"1", "true", "yes", "on", "trip", "tripped", "stop", "kill"})
"""Environment values that trip the switch (case-insensitive)."""

FALSE_VALUES: frozenset[str] = frozenset({"0", "false", "no", "off", "clear", ""})
"""Environment values that abstain. These do *not* clear a file sentinel."""

DEFAULT_ENV_VAR = "STAGEGATE_KILL"
DEFAULT_PATH_ENV_VAR = "STAGEGATE_KILL_FILE"


@dataclass(frozen=True, slots=True)
class KillSwitchState:
    """A point-in-time reading of the switch.

    Attributes:
        tripped: Whether execution must be suppressed.
        source: Which check decided: ``"env"``, ``"file"``, ``"probe"``,
            ``"error"``, or ``"clear"``.
        reason: Human-readable explanation, recorded in the audit log.
        checked_at: ``time.time()`` of the reading.
    """

    tripped: bool
    source: str
    reason: str | None = None
    checked_at: float = 0.0

    def to_record(self) -> dict[str, object]:
        """Audit-log representation."""
        return {"tripped": self.tripped, "source": self.source, "reason": self.reason}


class KillSwitch:
    """Checks whether the agent is currently allowed to act.

    Args:
        path: File sentinel path. Defaults to ``$STAGEGATE_KILL_FILE``; when that
            is unset too, the file check is skipped and only the environment
            variable is consulted.
        env_var: Environment variable name to consult.
        extra_probes: Additional named checks, run after env and file. Each
            returns ``(tripped, reason)``. A probe that raises trips the switch.
            Use for a feature-flag service or a control-plane heartbeat -- but see
            ``cache_ttl``, because a probe runs on every guarded call.
        cache_ttl: Seconds to reuse a reading. Defaults to ``0.0``: check every
            time. A stale *clear* reading is precisely the failure this component
            exists to prevent, so caching is opt-in. A local ``stat`` costs a few
            microseconds; set a TTL only when you have measured that it matters,
            or when a probe makes a network call.
        read_reason: Read the sentinel's first line as the reason. Set ``False``
            if the sentinel may live somewhere you would rather not read from.

    Example:
        >>> switch = KillSwitch(path="/var/run/stagegate/kill")
        >>> switch.check().tripped
        False
    """

    def __init__(
        self,
        path: str | os.PathLike[str] | None = None,
        *,
        env_var: str = DEFAULT_ENV_VAR,
        extra_probes: dict[str, Callable[[], tuple[bool, str | None]]] | None = None,
        cache_ttl: float = 0.0,
        read_reason: bool = True,
    ) -> None:
        self.env_var = env_var
        self.extra_probes = dict(extra_probes or {})
        self.cache_ttl = max(0.0, float(cache_ttl))
        self.read_reason = read_reason
        self._explicit_path = path is not None
        self._path = Path(path) if path is not None else None
        self._lock = threading.Lock()
        self._cached: KillSwitchState | None = None

    @property
    def path(self) -> Path | None:
        """Sentinel path in force, resolving ``$STAGEGATE_KILL_FILE`` each read.

        Resolved lazily rather than at construction so that a process which sets
        the variable after import still gets the sentinel it asked for.
        """
        if self._explicit_path:
            return self._path
        from_env = os.environ.get(DEFAULT_PATH_ENV_VAR)
        return Path(from_env) if from_env else None

    def check(self) -> KillSwitchState:
        """Read the switch, honouring the documented precedence and any TTL."""
        now = time.time()
        if self.cache_ttl > 0.0:
            with self._lock:
                cached = self._cached
                if cached is not None and (now - cached.checked_at) < self.cache_ttl:
                    return cached
        state = self._check_uncached(now)
        if self.cache_ttl > 0.0:
            with self._lock:
                self._cached = state
        return state

    def _check_uncached(self, now: float) -> KillSwitchState:
        # 1. Environment variable.
        raw = os.environ.get(self.env_var)
        if raw is not None:
            value = raw.strip().lower()
            if value in TRUE_VALUES:
                return KillSwitchState(True, "env", f"{self.env_var}={raw!r}", now)
            if value not in FALSE_VALUES:
                return KillSwitchState(
                    True,
                    "env",
                    f"{self.env_var}={raw!r} is not a recognised value; failing closed",
                    now,
                )
            # Recognised false: abstain, fall through to the sentinel.

        # 2. File sentinel.
        path = self.path
        if path is not None:
            try:
                exists = path.exists()
            except OSError as exc:
                return KillSwitchState(
                    True, "error", f"cannot stat kill-switch sentinel {path}: {exc}", now
                )
            if exists:
                return KillSwitchState(True, "file", self._sentinel_reason(path), now)

        # 3. Extra probes.
        for name, probe in self.extra_probes.items():
            try:
                tripped, reason = probe()
            except Exception as exc:  # noqa: BLE001 - a probe that throws fails closed
                return KillSwitchState(
                    True, "probe", f"kill-switch probe {name!r} raised {type(exc).__name__}: {exc}", now
                )
            if tripped:
                return KillSwitchState(True, "probe", reason or f"probe {name!r} tripped", now)

        return KillSwitchState(False, "clear", None, now)

    def _sentinel_reason(self, path: Path) -> str:
        base = f"kill-switch sentinel present at {path}"
        if not self.read_reason:
            return base
        try:
            with open(path, encoding="utf-8", errors="replace") as handle:
                first = handle.readline(512).strip()
        except OSError:
            return base
        return f"{base}: {first}" if first else base

    def trip(self, reason: str = "tripped by stagegate") -> Path:
        """Create the sentinel file. Convenience for tests and operator tooling.

        Raises:
            RuntimeError: if no sentinel path is configured -- there is nothing to
                write, and silently doing nothing would be the worst outcome for a
                method named ``trip``.
        """
        path = self.path
        if path is None:
            raise RuntimeError(
                "no kill-switch sentinel path configured; pass path=... or set "
                f"{DEFAULT_PATH_ENV_VAR}"
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(reason + "\n", encoding="utf-8")
        self.invalidate()
        return path

    def clear(self) -> None:
        """Remove the sentinel file if present. Convenience for tests and tooling."""
        path = self.path
        if path is not None:
            try:
                path.unlink()
            except FileNotFoundError:
                pass
        self.invalidate()

    def invalidate(self) -> None:
        """Drop any cached reading, forcing the next :meth:`check` to be fresh."""
        with self._lock:
            self._cached = None
