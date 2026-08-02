"""Evaluate one shared protocol-corpus case with this repository's parsers.

The third review found the two repositories each keeping a private
"compatible" fixture, which proves only that each parser agrees with itself.
There is now one corpus, byte-identical in both trees, and both sides run
every case through their own parser and must reach the same classified
verdict.

This module deliberately contains no fixtures. It maps a case's declared
`kind` onto the production parser for that kind, so a corpus case exercises
the same code a live campaign does.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from gladius_vllm.evidence_codes import (
    APPLICATION_SCHEMA_INVALID,
    RECEIPT_EXPECTATION_INCOMPLETE,
    RECEIPT_SCHEMA_INVALID,
    SEAL_REQUEST_SCHEMA_INVALID,
    SEAL_SCHEMA_INVALID,
    TELEMETRY_SCHEMA_INVALID,
)

CORPUS_FILENAME = "gladius-third-review-corpus.json"
CORPUS_VERSION = "gladius-third-review-corpus-v1"


@dataclass(frozen=True)
class CorpusVerdict:
    accepted: bool
    error_code: str | None
    detail: str | None


def _parsers() -> dict[str, tuple[Callable[[Any], object], str]]:
    """Production parser and default refusal code, per corpus case kind."""
    from gladius_vllm.application import parse_policy_application_v2
    from gladius_vllm.receipt import DeploymentExpectation, parse_server_start_receipt
    from gladius_vllm.seal_lifecycle import parse_seal_request
    from gladius_vllm.telemetry import parse_telemetry_record_v2, parse_telemetry_seal

    return {
        "server_start_receipt": (parse_server_start_receipt, RECEIPT_SCHEMA_INVALID),
        "policy_application": (
            parse_policy_application_v2,
            APPLICATION_SCHEMA_INVALID,
        ),
        "telemetry_record": (parse_telemetry_record_v2, TELEMETRY_SCHEMA_INVALID),
        "telemetry_seal": (parse_telemetry_seal, SEAL_SCHEMA_INVALID),
        "deployment_manifest": (
            DeploymentExpectation.from_dict,
            RECEIPT_EXPECTATION_INCOMPLETE,
        ),
        "seal_request": (parse_seal_request, SEAL_REQUEST_SCHEMA_INVALID),
    }


def evaluate_corpus_case(case: dict[str, Any]) -> CorpusVerdict:
    """Run one corpus case through the production parser for its kind.

    A refusal's code is the classified prefix the parser itself raised when
    it has one; otherwise it is the schema-invalid code for that kind. That
    distinction matters: a case declaring a *specific* violated invariant
    must be refused for that reason and not merely refused.
    """
    parsers = _parsers()
    kind = case.get("kind")
    if kind not in parsers:
        raise ValueError(f"unknown corpus case kind {kind!r}")
    parser, default_code = parsers[kind]

    try:
        parser(case["payload"])
    except Exception as error:  # noqa: BLE001 - any refusal is a refusal
        message = str(error)
        code, separator, _ = message.partition(":")
        classified = code.strip() if separator and code.strip().isupper() else None
        return CorpusVerdict(
            accepted=False,
            error_code=classified or default_code,
            detail=message,
        )
    return CorpusVerdict(accepted=True, error_code=None, detail=None)
