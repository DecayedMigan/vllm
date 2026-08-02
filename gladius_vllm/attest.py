"""`python -m gladius_vllm.attest` -- publish and verify a server-start receipt.

Run by the launcher once the API socket is bound. It contributes only the
API-process facts it can prove locally (which PID owns the listen socket, and
that PID's reuse-proof start identity) and joins them with the contribution
the EngineCore process already published. It never measures the model,
tokenizer, GPU, or itself: those are EngineCore-side facts by construction.

    python -m gladius_vllm.attest publish \\
        --policy-dir /run/gladius/gpu0 --nonce "$GLADIUS_ATTESTATION_NONCE" \\
        --host 127.0.0.1 --port 8000

    python -m gladius_vllm.attest verify \\
        --policy-dir /run/gladius/gpu0 --nonce "$GLADIUS_ATTESTATION_NONCE" \\
        --host 127.0.0.1 --port 8000 --expect-gpu-uuid GPU-... --formal
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from gladius_vllm.digest import sha256_file
from gladius_vllm.netowner import SocketOwnerError, listen_socket_owner_pid
from gladius_vllm.receipt import (
    DeploymentExpectation,
    ReceiptError,
    assemble_server_start_receipt,
    read_server_start_receipt,
    verify_receipt_against_deployment,
)
from gladius_vllm.seal_lifecycle import (
    ACK_SEALED,
    SealRequestError,
    read_seal_ack,
    write_seal_request,
)

_EXIT_OK = 0
_EXIT_INVALID = 1


def _wait_for_contribution(policy_dir: Path, timeout_seconds: float) -> None:
    from gladius_vllm.receipt import ENGINE_CONTRIBUTION_FILENAME

    deadline = time.monotonic() + timeout_seconds
    path = policy_dir / ENGINE_CONTRIBUTION_FILENAME
    while True:
        if path.is_file():
            return
        if time.monotonic() >= deadline:
            raise ReceiptError(
                f"EngineCore did not publish {ENGINE_CONTRIBUTION_FILENAME} "
                f"within {timeout_seconds:g}s"
            )
        time.sleep(0.1)


def _seal(args: argparse.Namespace, policy_dir: Path) -> int:
    """Ask the live server to seal, then wait for *its* acknowledgement.

    The operator never writes the seal. That is the whole point: a seal
    produced outside the serving process would certify a stream the server
    might still be appending to, and would prove nothing about which process
    produced it.
    """
    receipt = read_server_start_receipt(policy_dir / "server_start_receipt.json")
    expectation = DeploymentExpectation.from_file(args.deployment_manifest)
    errors = verify_receipt_against_deployment(receipt, expectation)
    if errors:
        print(
            json.dumps({"valid": False, "errors": errors}, indent=2, sort_keys=True),
            file=sys.stderr,
        )
        return _EXIT_INVALID

    write_seal_request(
        policy_dir,
        server_instance_id=receipt.server_instance_id,
        deployment_manifest_sha256=sha256_file(args.deployment_manifest),
        expected_final_generation=args.expect_final_generation,
        attestation_nonce=args.nonce,
    )

    deadline = time.monotonic() + args.wait_seconds
    while True:
        ack = read_seal_ack(policy_dir)
        # An acknowledgement for a different instance is somebody else's
        # answer, so it is not this request's answer.
        if ack is not None and ack["server_instance_id"] == receipt.server_instance_id:
            print(json.dumps(ack, indent=2, sort_keys=True))
            return _EXIT_OK if ack["status"] == ACK_SEALED else _EXIT_INVALID
        if time.monotonic() >= deadline:
            raise ReceiptError(
                f"the server did not acknowledge the seal request within "
                f"{args.wait_seconds:g}s; if it is idle it may need one "
                "discarded request to reach its next scheduling boundary"
            )
        time.sleep(0.1)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gladius_vllm.attest")
    subparsers = parser.add_subparsers(dest="command", required=True)

    seal = subparsers.add_parser(
        "seal",
        help=(
            "ask the live server to certify and retire its policy directory, "
            "then wait for its instance-bound acknowledgement"
        ),
    )
    seal.add_argument("--policy-dir", type=Path, required=True)
    seal.add_argument("--nonce", required=True)
    seal.add_argument("--deployment-manifest", type=Path, required=True)
    seal.add_argument(
        "--expect-final-generation",
        type=int,
        default=None,
        help="refuse to seal before the server has reached this generation",
    )
    seal.add_argument("--wait-seconds", type=float, default=300.0)

    for name in ("publish", "verify"):
        subparser = subparsers.add_parser(name)
        subparser.add_argument("--policy-dir", type=Path, required=True)
        subparser.add_argument("--nonce", required=True)
        subparser.add_argument("--host", required=True)
        subparser.add_argument("--port", type=int, required=True)
        if name == "publish":
            subparser.add_argument(
                "--expect-api-pid",
                type=int,
                default=None,
                help=(
                    "cross-check only: the PID is always derived from the "
                    "listening socket, and publication fails if this disagrees"
                ),
            )
            subparser.add_argument("--wait-seconds", type=float, default=300.0)
        else:
            # Required, not optional. There is exactly one verification
            # path and it consumes the whole expectation: an argument that
            # can be omitted is a check that can be skipped, and the
            # reviewed revision reported skipped checks as passes.
            subparser.add_argument(
                "--deployment-manifest",
                type=Path,
                required=True,
                help="the immutable manifest of expected identity and digests",
            )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    policy_dir = Path(args.policy_dir)

    try:
        if args.command == "seal":
            return _seal(args, policy_dir)
        if args.command == "publish":
            _wait_for_contribution(policy_dir, args.wait_seconds)
            # Always derived from the bound socket, never accepted from the
            # caller. A supplied PID that was used directly (the reviewed
            # behaviour) could bind a live but unrelated process to the
            # endpoint the campaign will actually call.
            api_pid = listen_socket_owner_pid(args.host, args.port)
            if args.expect_api_pid is not None and args.expect_api_pid != api_pid:
                raise ReceiptError(
                    f"{args.host}:{args.port} is owned by PID {api_pid}, not the "
                    f"expected {args.expect_api_pid}"
                )
            receipt = assemble_server_start_receipt(
                policy_dir,
                api_pid=api_pid,
                listen_host=args.host,
                listen_port=args.port,
                expected_nonce=args.nonce,
            )
            print(
                json.dumps(
                    {
                        "server_instance_id": receipt.server_instance_id,
                        "engine_id": receipt.engine_id,
                        "model_id": receipt.model_id,
                        "physical_gpu_uuid": receipt.physical_gpu_uuid,
                        "api_pid": receipt.api_pid,
                        "engine_core_pid": receipt.engine_core_pid,
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return _EXIT_OK

        receipt = read_server_start_receipt(policy_dir / "server_start_receipt.json")
        expectation = DeploymentExpectation.from_file(args.deployment_manifest)
        for name, supplied, expected in (
            ("nonce", args.nonce, expectation.attestation_nonce),
            ("host", args.host, expectation.listen_host),
            ("port", args.port, expectation.listen_port),
        ):
            if supplied != expected:
                raise ReceiptError(
                    f"--{name} {supplied!r} disagrees with the deployment "
                    f"manifest's {expected!r}"
                )
        # The same function SMIG and the contract tests call. One
        # implementation means the two repositories cannot drift into
        # disagreeing about what a valid receipt is.
        errors = verify_receipt_against_deployment(receipt, expectation)
    except (ReceiptError, SealRequestError, SocketOwnerError) as error:
        print(f"attestation failed: {error}", file=sys.stderr)
        return _EXIT_INVALID
    except (OSError, ValueError) as error:
        print(f"attestation unreadable: {error}", file=sys.stderr)
        return _EXIT_INVALID

    print(
        json.dumps(
            {
                "server_instance_id": receipt.server_instance_id,
                "valid": not errors,
                "errors": errors,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return _EXIT_OK if not errors else _EXIT_INVALID


if __name__ == "__main__":  # pragma: no cover - module entry point
    raise SystemExit(main())
