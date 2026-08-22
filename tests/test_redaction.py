"""Redaction: what reaches a sink, and what must never."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from stagegate import (
    REDACTED,
    Outcome,
    RedactionError,
    RedactionPolicy,
    Stage,
    StageGate,
    StaticApprovalHandler,
)
from stagegate.redaction import normalise_key


@pytest.mark.parametrize(
    "key",
    ["password", "API_KEY", "api-key", "apiKey", "Authorization", "client_secret",
     "refresh_token", "ssn", "credit_card", "email", "phone_number", "iban"],
)
def test_known_secret_and_pii_key_names_are_redacted(key: str) -> None:
    assert RedactionPolicy()({key: "sensitive"})[key] == REDACTED


def test_naming_style_does_not_defeat_a_rule() -> None:
    assert normalise_key("API_Key") == normalise_key("api-key") == normalise_key("apiKey")


def test_ordinary_arguments_pass_through_unchanged() -> None:
    policy = RedactionPolicy()
    assert policy({"ticket_id": "T-1001", "count": 3, "ok": True, "ratio": 0.5}) == {
        "ticket_id": "T-1001", "count": 3, "ok": True, "ratio": 0.5,
    }


def test_nested_mappings_are_walked() -> None:
    result = RedactionPolicy()({"config": {"host": "db-1", "password": "hunter2"}})
    assert result["config"] == {"host": "db-1", "password": REDACTED}


def test_values_inside_lists_are_walked() -> None:
    result = RedactionPolicy()({"users": [{"name": "ada", "ssn": "123-45-6789"}]})
    assert result["users"][0] == {"name": "ada", "ssn": REDACTED}


@pytest.mark.parametrize(
    "value",
    [
        "-----BEGIN RSA PRIVATE KEY-----\nMIIE...",
        "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dBjftJeZ4CVP",
        "AKIAIOSFODNN7EXAMPLE",
        "sk-abcdefghijklmnopqrstuvwxyz",
        "ghp_abcdefghijklmnopqrstuvwxyz0123",
        "078-05-1120",
    ],
)
def test_secret_shaped_values_are_caught_under_an_innocent_name(value: str) -> None:
    assert RedactionPolicy()({"note": value})["note"] == REDACTED


@pytest.mark.parametrize(
    "value",
    ["ticket T-1001 is blocked", "version 1.2.3", "call me on extension 4471", "sk-short"],
)
def test_ordinary_text_is_not_falsely_redacted(value: str) -> None:
    """A policy that redacts too eagerly gets switched off, which is worse."""
    assert RedactionPolicy()({"note": value})["note"] == value


def test_a_cycle_does_not_hang_or_blow_the_stack() -> None:
    payload: dict = {"name": "loop"}
    payload["self"] = payload
    result = RedactionPolicy()({"payload": payload})
    assert result["payload"]["self"] == "[CYCLE]"


def test_deep_nesting_is_bounded() -> None:
    deep: dict = {"leaf": 1}
    for _ in range(20):
        deep = {"next": deep}
    rendered = str(RedactionPolicy(max_depth=3)({"deep": deep}))
    assert "[MAX_DEPTH]" in rendered


def test_long_strings_are_truncated_and_say_how_much_was_withheld() -> None:
    result = RedactionPolicy(max_string=20)({"body": "x" * 100})["body"]
    assert result.startswith("x" * 20)
    assert "+80 chars" in result


def test_long_collections_are_capped_not_silently_dropped() -> None:
    policy = RedactionPolicy(max_items=3)
    listed = policy({"items": list(range(10))})["items"]
    assert listed[:3] == [0, 1, 2]
    assert "+7 more items" in listed[-1]

    mapped = policy({"m": {f"k{i}": i for i in range(10)}})["m"]
    assert mapped["[TRUNCATED]"] == "+7 more keys"


def test_bytes_are_summarised_not_dumped() -> None:
    assert RedactionPolicy()({"blob": b"\x00\x01\x02"})["blob"] == "<bytes len=3>"


def test_dataclass_arguments_are_walked_by_field_name() -> None:
    @dataclass
    class Credentials:
        username: str
        password: str

    result = RedactionPolicy()({"creds": Credentials("ada", "hunter2")})
    assert result["creds"] == {"username": "ada", "password": REDACTED}


def test_an_object_whose_repr_raises_does_not_break_the_call() -> None:
    class Hostile:
        def __repr__(self) -> str:
            raise RuntimeError("no repr for you")

    assert RedactionPolicy()({"thing": Hostile()})["thing"] == "[UNSERIALISABLE]"


def test_non_json_floats_are_made_serialisable() -> None:
    result = RedactionPolicy()({"a": float("nan"), "b": float("inf")})
    assert result == {"a": "nan", "b": "inf"}


def test_allow_keys_is_the_escape_hatch() -> None:
    policy = RedactionPolicy(allow_keys={"token_count"})
    assert policy({"token_count": 512, "token": "abc"}) == {"token_count": 512, "token": REDACTED}


def test_key_patterns_cover_families_of_names() -> None:
    import re

    policy = RedactionPolicy(keys=frozenset(), key_patterns=(re.compile(r"^x.*header$"),))
    assert policy({"x_auth_header": "abc", "body": "fine"}) == {"x_auth_header": REDACTED, "body": "fine"}


def test_secrets_only_leaves_personal_data_readable() -> None:
    result = RedactionPolicy.secrets_only()({"email": "ada@example.com", "password": "hunter2"})
    assert result == {"email": "ada@example.com", "password": REDACTED}


def test_disabled_still_bounds_and_normalises() -> None:
    policy = RedactionPolicy.disabled()
    assert policy({"password": "hunter2"}) == {"password": "hunter2"}
    payload: dict = {}
    payload["self"] = payload
    assert policy({"p": payload})["p"]["self"] == "[CYCLE]"


# ------------------------------------------------------------ fingerprints


def test_fingerprinting_without_a_salt_is_refused() -> None:
    """An unsalted digest of an SSN is recoverable in seconds."""
    with pytest.raises(RedactionError, match="needs a salt"):
        RedactionPolicy(fingerprint=True)


def test_fingerprints_correlate_without_revealing(monkeypatch: pytest.MonkeyPatch) -> None:
    policy = RedactionPolicy(fingerprint=True, salt="deployment-salt")
    same_a = policy({"password": "hunter2"})["password"]
    same_b = policy({"password": "hunter2"})["password"]
    other = policy({"password": "different"})["password"]

    assert same_a == same_b
    assert same_a != other
    assert "hunter2" not in same_a
    assert same_a.startswith("[REDACTED:fp:")


def test_the_salt_can_come_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STAGEGATE_REDACTION_SALT", "from-env")
    assert RedactionPolicy(fingerprint=True)({"password": "x"})["password"].startswith("[REDACTED:fp:")


def test_a_different_salt_gives_different_fingerprints() -> None:
    a = RedactionPolicy(fingerprint=True, salt="salt-a")({"token": "abc"})["token"]
    b = RedactionPolicy(fingerprint=True, salt="salt-b")({"token": "abc"})["token"]
    assert a != b


# ----------------------------------------------------- integrated with the gate


def test_arguments_are_redacted_before_they_reach_the_sink(sink, calls) -> None:
    gate = StageGate(audit=sink)

    @gate.capability("demo.login", stage=Stage.ACT)
    def login(username: str, password: str) -> None:
        calls.append(username)

    login("ada", password="hunter2")

    assert sink.events[0].arguments == {"username": "ada", "password": REDACTED}
    assert "hunter2" not in str(sink.records[0])


def test_a_per_capability_policy_overrides_the_gate_default(sink) -> None:
    gate = StageGate(audit=sink, redaction=RedactionPolicy())

    @gate.capability("demo.support", stage=Stage.ACT, redact=RedactionPolicy.secrets_only())
    def support(email: str, token: str) -> None: ...

    support("ada@example.com", token="abc")
    assert sink.events[0].arguments == {"email": "ada@example.com", "token": REDACTED}


def test_the_effect_description_never_sees_raw_arguments(sink) -> None:
    """The invariant: no raw value leaves the process through StageGate."""
    gate = StageGate(audit=sink)

    @gate.capability("demo.auth", stage=Stage.OBSERVE, describe="authenticate with {password}")
    def auth(password: str) -> None: ...

    auth("hunter2")

    assert sink.events[0].effect == f"authenticate with {REDACTED}"
    assert "hunter2" not in str(sink.records[0])


def test_the_approver_sees_the_same_redacted_view_as_the_log(sink) -> None:
    handler = StaticApprovalHandler(True)
    gate = StageGate(audit=sink, approval=handler)

    @gate.capability("demo.cap", stage=Stage.SUGGEST)
    def cap(user: str, api_key: str) -> None: ...

    cap("ada", api_key="sk-abcdefghijklmnopqrstuv")

    assert handler.requests[0].arguments == {"user": "ada", "api_key": REDACTED}


def test_a_redactor_that_raises_withholds_everything(sink, calls) -> None:
    """Fail closed on the arguments; you lose the values, not the event."""
    def broken(arguments):
        raise ValueError("policy is misconfigured")

    gate = StageGate(audit=sink, redaction=broken)

    @gate.capability("demo.act", stage=Stage.ACT)
    def act(secret_value: str) -> None:
        calls.append("ran")

    result = act("do-not-log-me")

    assert calls == ["ran"], "the call still happens; only the arguments are lost"
    assert "do-not-log-me" not in str(sink.records[0])
    assert sink.events[0].arguments == {"[REDACTION_FAILED]": ["secret_value"]}
    assert "redaction policy raised" in (sink.events[0].decision_note or "")


def test_an_exception_message_containing_a_secret_is_scrubbed(sink) -> None:
    gate = StageGate(audit=sink)

    @gate.capability("demo.act", stage=Stage.ACT, propagate_errors=False)
    def act() -> None:
        raise ValueError("rejected token sk-abcdefghijklmnopqrstuv")

    result = act()
    assert result.outcome is Outcome.FAILED
    assert "sk-abcdefghijkl" not in str(sink.records[0])


def test_a_plain_function_is_a_valid_redactor(sink) -> None:
    gate = StageGate(audit=sink, redaction=lambda args: {k: "***" for k in args})

    @gate.capability("demo.act", stage=Stage.ACT)
    def act(a: str, b: str) -> None: ...

    act("one", "two")
    assert sink.events[0].arguments == {"a": "***", "b": "***"}


def test_a_broken_describer_does_not_break_the_call(sink, calls) -> None:
    gate = StageGate(audit=sink)

    @gate.capability("demo.act", stage=Stage.ACT, describe="needs {missing_field}")
    def act(present: str) -> None:
        calls.append("ran")

    result = act("x")
    assert result.outcome is Outcome.EXECUTED
    assert "description unavailable" in sink.events[0].effect
