"""Redaction hooks, so arguments can be audited without leaking what they carry.

An audit log is only useful if people are willing to keep it. The moment it
contains bearer tokens or customer identifiers, it acquires the handling
requirements of the most sensitive thing in it, and in practice that means
someone turns it off. Redaction runs *before* anything reaches a sink, so the
sensitive value never exists in the record at all.

Two mechanisms, applied together:

* **Key names.** Any argument (or nested mapping key) whose normalised name
  matches a known secret or PII name is replaced. Normalisation lowercases and
  drops ``_``, ``-`` and spaces, so ``API_KEY``, ``api-key`` and ``apiKey`` all
  match the same rule.
* **Value shapes.** A small set of high-precision patterns -- PEM private key
  headers, JWTs, AWS access key ids, ``sk-``-style API keys, US SSNs -- catch
  secrets that arrive under an innocent name.

A redactor is any callable taking the argument mapping and returning a
sanitised one, so ``redact=lambda args: {...}`` is a legitimate redactor.
:class:`RedactionPolicy` is the batteries-included implementation.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import re
from collections.abc import Mapping, Sequence, Set
from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any, Protocol, runtime_checkable

from .errors import RedactionError

__all__ = [
    "Redactor",
    "RedactionPolicy",
    "DEFAULT_SECRET_KEYS",
    "DEFAULT_PII_KEYS",
    "DEFAULT_VALUE_PATTERNS",
    "REDACTED",
    "normalise_key",
]

REDACTED = "[REDACTED]"
"""Placeholder substituted for a redacted value."""

_TRUNCATED = "[TRUNCATED]"
_UNSERIALISABLE = "[UNSERIALISABLE]"
_CYCLE = "[CYCLE]"
_DEPTH = "[MAX_DEPTH]"

DEFAULT_SECRET_KEYS: frozenset[str] = frozenset(
    {
        "password", "passwd", "pwd", "secret", "secrets", "token", "accesstoken",
        "refreshtoken", "idtoken", "apikey", "apisecret", "authorization", "auth",
        "credential", "credentials", "privatekey", "secretkey", "clientsecret",
        "sessionkey", "sessionid", "cookie", "bearer", "signature", "otp", "mfacode",
        "pin", "passphrase", "salt",
    }
)
"""Argument names treated as credentials. Normalised form (see :func:`normalise_key`)."""

DEFAULT_PII_KEYS: frozenset[str] = frozenset(
    {
        "ssn", "socialsecuritynumber", "nationalid", "taxid", "ein", "dob",
        "dateofbirth", "birthdate", "creditcard", "cardnumber", "pan", "cvv", "cvc",
        "accountnumber", "routingnumber", "iban", "bic", "swift", "email",
        "emailaddress", "phone", "phonenumber", "mobile", "address", "streetaddress",
        "postaladdress", "zipcode", "postcode", "passportnumber", "driverslicense",
        "medicalrecordnumber", "mrn", "policynumber",
    }
)
"""Argument names treated as personal data. Normalised form."""

DEFAULT_VALUE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("pem_private_key", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    ("jwt", re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{4,}\b")),
    ("aws_access_key_id", re.compile(r"\b(?:AKIA|ASIA|AIDA|AROA)[0-9A-Z]{16}\b")),
    ("api_key_prefix", re.compile(r"\b(?:sk|rk|pk)-[A-Za-z0-9_-]{16,}\b")),
    ("github_token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b")),
    ("us_ssn", re.compile(r"\b(?!000|666|9\d\d)\d{3}-(?!00)\d{2}-(?!0000)\d{4}\b")),
)
"""High-precision value shapes. Chosen for near-zero false positives: a policy
that redacts too eagerly gets disabled, which is worse than one that misses."""

_KEY_STRIP = re.compile(r"[\s_\-.]+")


def normalise_key(key: str) -> str:
    """Lowercase ``key`` and strip separators, so naming style does not defeat a rule.

    >>> normalise_key("API_Key"), normalise_key("api-key"), normalise_key("apiKey")
    ('apikey', 'apikey', 'apikey')
    """
    return _KEY_STRIP.sub("", str(key)).lower()


@runtime_checkable
class Redactor(Protocol):
    """Anything that sanitises an argument mapping for logging.

    A plain function satisfies this, which is the point: reaching for a custom
    redactor should not require subclassing anything.
    """

    def __call__(self, arguments: Mapping[str, Any]) -> Mapping[str, Any]:
        """Return a sanitised, JSON-friendly copy of ``arguments``."""
        ...


@dataclass(frozen=True)
class RedactionPolicy:
    """Recursive, cycle-safe, size-bounded redaction of a call's arguments.

    Args:
        keys: Normalised argument names to redact. Defaults to secrets + PII.
        key_patterns: Regexes matched against the *normalised* key, for families
            of names (``re.compile(r"^x.*header$")``).
        value_patterns: ``(label, regex)`` pairs matched against string values.
        allow_keys: Normalised names that are never redacted, even if a rule
            matches. Escape hatch for a field called ``token_count``.
        placeholder: What replaces a redacted value.
        max_depth: Nesting depth beyond which structures collapse to ``[MAX_DEPTH]``.
        max_items: Per-collection element cap. Excess is summarised, not dropped
            silently.
        max_string: Strings longer than this are truncated with a marker naming
            the number of characters withheld.
        fingerprint: Emit ``[REDACTED:fp:<12 hex>]`` instead of a bare placeholder,
            so an auditor can tell whether two calls used the *same* secret without
            learning it. Requires ``salt``.
        salt: HMAC key for fingerprints. Falls back to ``STAGEGATE_REDACTION_SALT``.

    Raises:
        RedactionError: if ``fingerprint`` is on but no salt is available. An
            unsalted digest of a low-entropy value (an SSN, a phone number) is
            recoverable by brute force in seconds, so this refuses rather than
            hands out a false sense of safety.

    Example:
        >>> policy = RedactionPolicy()
        >>> policy({"user": "ada", "api_key": "sk-abcdefghijklmnopqrstuv"})
        {'user': 'ada', 'api_key': '[REDACTED]'}
    """

    keys: frozenset[str] = field(default=DEFAULT_SECRET_KEYS | DEFAULT_PII_KEYS)
    key_patterns: tuple[re.Pattern[str], ...] = ()
    value_patterns: tuple[tuple[str, re.Pattern[str]], ...] = DEFAULT_VALUE_PATTERNS
    allow_keys: frozenset[str] = frozenset()
    placeholder: str = REDACTED
    max_depth: int = 6
    max_items: int = 50
    max_string: int = 512
    fingerprint: bool = False
    salt: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "keys", frozenset(normalise_key(k) for k in self.keys))
        object.__setattr__(self, "allow_keys", frozenset(normalise_key(k) for k in self.allow_keys))
        if self.fingerprint and not self._salt_bytes():
            raise RedactionError(
                "RedactionPolicy(fingerprint=True) needs a salt: pass salt=... or set "
                "STAGEGATE_REDACTION_SALT. Unsalted digests of low-entropy values "
                "(SSNs, phone numbers, short ids) are trivially reversible."
            )

    @classmethod
    def secrets_only(cls, **overrides: Any) -> RedactionPolicy:
        """A policy that redacts credentials but leaves personal data readable.

        Appropriate when the audit log is already inside the data boundary that
        governs the personal data in question, and analysts need to read it.
        """
        return cls(keys=DEFAULT_SECRET_KEYS, **overrides)

    @classmethod
    def disabled(cls) -> RedactionPolicy:
        """Structure-normalising only: no key or value rules, still size-bounded.

        Still worth using over raw ``repr``: it bounds sizes, survives cycles, and
        produces JSON-serialisable output.
        """
        return cls(keys=frozenset(), value_patterns=(), allow_keys=frozenset())

    def __call__(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        """Return a sanitised, JSON-serialisable copy of ``arguments``."""
        seen: set[int] = set()
        return {
            str(key): self._walk(value, str(key), 0, seen)
            for key, value in arguments.items()
        }

    def _salt_bytes(self) -> bytes | None:
        raw = self.salt if self.salt is not None else os.environ.get("STAGEGATE_REDACTION_SALT")
        if not raw:
            return None
        return raw.encode("utf-8")

    def _redacted(self, value: Any) -> str:
        if not self.fingerprint:
            return self.placeholder
        salt = self._salt_bytes()
        if salt is None:  # pragma: no cover - blocked in __post_init__
            return self.placeholder
        digest = hmac.new(salt, repr(value).encode("utf-8", "replace"), hashlib.sha256)
        return f"[REDACTED:fp:{digest.hexdigest()[:12]}]"

    def _should_redact_key(self, key: str) -> bool:
        norm = normalise_key(key)
        if norm in self.allow_keys:
            return False
        if norm in self.keys:
            return True
        return any(pattern.search(norm) for pattern in self.key_patterns)

    def _scrub_string(self, text: str) -> str:
        """Apply value patterns, then bound the length."""
        for _label, pattern in self.value_patterns:
            if pattern.search(text):
                return self._redacted(text)
        if len(text) > self.max_string:
            withheld = len(text) - self.max_string
            return f"{text[: self.max_string]}{_TRUNCATED}(+{withheld} chars)"
        return text

    def _walk(self, value: Any, key: str, depth: int, seen: set[int]) -> Any:
        if self._should_redact_key(key):
            return self._redacted(value)
        return self._walk_value(value, depth, seen)

    def _walk_value(self, value: Any, depth: int, seen: set[int]) -> Any:
        if value is None or isinstance(value, (bool, int, float)):
            # bool before int matters only for JSON fidelity; both are safe as-is.
            if isinstance(value, float) and (value != value or value in (float("inf"), float("-inf"))):
                return repr(value)  # NaN/Infinity are not valid JSON
            return value
        if isinstance(value, str):
            return self._scrub_string(value)
        if isinstance(value, (bytes, bytearray, memoryview)):
            return f"<{type(value).__name__} len={len(bytes(value))}>"

        if depth >= self.max_depth:
            return _DEPTH

        marker = id(value)
        if marker in seen:
            return _CYCLE
        seen.add(marker)
        try:
            return self._walk_container(value, depth, seen)
        finally:
            seen.discard(marker)

    def _walk_container(self, value: Any, depth: int, seen: set[int]) -> Any:
        if isinstance(value, Mapping):
            out: dict[str, Any] = {}
            for index, (raw_key, item) in enumerate(value.items()):
                if index >= self.max_items:
                    out["[TRUNCATED]"] = f"+{len(value) - self.max_items} more keys"
                    break
                out[str(raw_key)] = self._walk(item, str(raw_key), depth + 1, seen)
            return out

        if is_dataclass(value) and not isinstance(value, type):
            return {
                f.name: self._walk(getattr(value, f.name, None), f.name, depth + 1, seen)
                for f in fields(value)
            }

        if isinstance(value, (Sequence, Set)) and not isinstance(value, (str, bytes, bytearray)):
            items = list(value)
            out_list = [self._walk_value(item, depth + 1, seen) for item in items[: self.max_items]]
            if len(items) > self.max_items:
                out_list.append(f"[TRUNCATED](+{len(items) - self.max_items} more items)")
            return out_list

        # Unknown object: fall back to a bounded repr, itself scrubbed. Never let
        # a __repr__ that raises take down the caller's capability.
        try:
            text = repr(value)
        except Exception:
            return _UNSERIALISABLE
        return self._scrub_string(text)


def _safe_text(text: str, policy: Redactor | None, limit: int = 2000) -> str:
    """Scrub a free-text string (an exception message, say) through ``policy``.

    Exception messages routinely embed the argument that caused them, so they get
    the same treatment as arguments before reaching a sink.
    """
    if not isinstance(text, str):
        text = str(text)
    text = text[:limit]
    if policy is None:
        return text
    try:
        scrubbed = policy({"message": text}).get("message", REDACTED)
    except Exception:
        return REDACTED
    return scrubbed if isinstance(scrubbed, str) else REDACTED
