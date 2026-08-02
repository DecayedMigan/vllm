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
import socket
import sys
import time
from pathlib import Path

from gladius_vllm.receipt import (
    ReceiptError,
    assemble_server_start_receipt,
    read_server_start_receipt,
    verify_server_start_receipt,
)

_EXIT_OK = 0
_EXIT_INVALID = 1


def _listen_socket_owner_pid(host: str, port: int) -> int:
    """Find the PID that holds the listening socket for `host:port`.

    Parsed from /proc rather than taken on trust from a `--api-pid` flag: the
    point of the API-side contribution is to prove that the attested process
    is the one actually serving the endpoint the campaign will call.
    """
    inodes = _listening_inodes(host, port)
    if not inodes:
        raise ReceiptError(f"no process is listening on {host}:{port}")
    for pid_dir in Path("/proc").iterdir():
        if not pid_dir.name.isdigit():
            continue
        fd_dir = pid_dir / "fd"
        try:
            descriptors = list(fd_dir.iterdir())
        except OSError:
            continue
        for descriptor in descriptors:
            try:
                target = descriptor.readlink().name
            except OSError:
                continue
            if target.startswith("socket:[") and target[8:-1] in inodes:
                return int(pid_dir.name)
    raise ReceiptError(
        f"the socket listening on {host}:{port} is not owned by a visible "
        "process; run the attestor as the same user as the API server"
    )


def _listening_inodes(host: str, port: int) -> set[str]:
    """Socket inodes in LISTEN state bound to `port` on a matching address."""
    wanted = {_hex_address(host, port), _hex_address("0.0.0.0", port)}
    if ":" not in host:
        wanted.add(_hex_address("::", port, ipv6=True))
    inodes: set[str] = set()
    for name, ipv6 in (("tcp", False), ("tcp6", True)):
        try:
            lines = Path(f"/proc/net/{name}").read_text().splitlines()[1:]
        except OSError:
            continue
        for line in lines:
            fields = line.split()
            if len(fields) < 10 or fields[3] != "0A":  # 0A == TCP_LISTEN
                continue
            local = fields[1].upper()
            if local in wanted or (
                ipv6 and local.endswith(f":{port:04X}") and _is_any_address(local)
            ):
                inodes.add(fields[9])
    return inodes


def _hex_address(host: str, port: int, *, ipv6: bool = False) -> str:
    if ipv6:
        packed = socket.inet_pton(socket.AF_INET6, host)
        words = [
            packed[index : index + 4][::-1].hex().upper() for index in (0, 4, 8, 12)
        ]
        return f"{''.join(words)}:{port:04X}"
    packed = socket.inet_pton(socket.AF_INET, host)
    return f"{packed[::-1].hex().upper()}:{port:04X}"


def _is_any_address(local: str) -> bool:
    return set(local.split(":")[0]) == {"0"}


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


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gladius_vllm.attest")
    subparsers = parser.add_subparsers(dest="command", required=True)

    for name in ("publish", "verify"):
        subparser = subparsers.add_parser(name)
        subparser.add_argument("--policy-dir", type=Path, required=True)
        subparser.add_argument("--nonce", required=True)
        subparser.add_argument("--host", required=True)
        subparser.add_argument("--port", type=int, required=True)
        if name == "publish":
            subparser.add_argument("--api-pid", type=int, default=None)
            subparser.add_argument("--wait-seconds", type=float, default=300.0)
        else:
            subparser.add_argument("--expect-gpu-uuid", default=None)
            subparser.add_argument("--expect-engine-id", default=None)
            subparser.add_argument("--expect-model-id", default=None)
            subparser.add_argument(
                "--formal",
                action="store_true",
                help="also require the frozen Qwen3-8B startup configuration",
            )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    policy_dir = Path(args.policy_dir)

    try:
        if args.command == "publish":
            _wait_for_contribution(policy_dir, args.wait_seconds)
            api_pid = args.api_pid or _listen_socket_owner_pid(args.host, args.port)
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
        errors = verify_server_start_receipt(
            receipt,
            expected_nonce=args.nonce,
            expected_engine_id=args.expect_engine_id,
            expected_model_id=args.expect_model_id,
            expected_listen_host=args.host,
            expected_listen_port=args.port,
            expected_gpu_uuid=args.expect_gpu_uuid,
            require_formal_startup=args.formal,
        )
    except ReceiptError as error:
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
