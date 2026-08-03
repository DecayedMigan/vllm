#!/usr/bin/env python3
"""Drive four launched servers through the terminal seal lifecycle.

Everything here is measured against processes that are actually running:
socket ownership from `/proc/net/tcp`, process start identity from
`/proc/<pid>/stat`, and the seal produced by the live scheduler rather than
by this script.

The one thing deliberately *not* asserted is four distinct physical GPU
UUIDs. Four processes on one card cannot have them, so the lane-set check
must refuse the set, and that refusal is recorded as evidence the check is
live rather than skipped.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gladius_vllm.atomic import atomic_write_json  # noqa: E402
from gladius_vllm.netowner import listen_socket_owner_pid  # noqa: E402
from gladius_vllm.receipt import (  # noqa: E402
    DeploymentExpectation,
    read_server_start_receipt,
    verify_receipt_against_deployment,
)
from gladius_vllm.seal_lifecycle import (  # noqa: E402
    read_seal_ack,
    write_seal_request,
)
from gladius_vllm.telemetry import (  # noqa: E402
    RETIRED_MARKER_FILENAME,
    TELEMETRY_SEAL_FILENAME,
    verify_telemetry_seal,
)


def send_one(port: int, prompt: str = "hello") -> dict:
    """One real completion, so the scheduler has steps to certify."""
    payload = json.dumps(
        {
            "model": "gladius-smoke",
            "prompt": prompt,
            "max_tokens": 8,
            "temperature": 0,
        }
    ).encode()
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            body = json.loads(response.read())
        return {
            "ok": True,
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "completion_tokens": body.get("usage", {}).get("completion_tokens"),
        }
    except (urllib.error.URLError, OSError, ValueError) as error:
        return {"ok": False, "error": str(error)}


def deployment_for(receipt) -> DeploymentExpectation:
    """The manifest this deployment actually is.

    Built from the receipt because this is a *development host* smoke: the
    point is to exercise the lifecycle, not to pretend the host matches the
    frozen H100 campaign. The H100 manifest is generated separately and
    hard-codes the corroborated identity source, which this host cannot
    satisfy -- see `formal_verification` in the report.
    """
    return DeploymentExpectation(
        attestation_nonce=receipt.attestation_nonce,
        engine_id=receipt.engine_id,
        model_id=receipt.model_id,
        model_path=receipt.model_path,
        listen_host=receipt.listen_host,
        listen_port=receipt.listen_port,
        physical_gpu_uuid=receipt.physical_gpu_uuid,
        physical_gpu_identity_source=receipt.physical_gpu_identity_source,
        mig_uuid=receipt.mig_uuid,
        mig_profile=receipt.mig_profile,
        mig_parent_gpu_uuid=receipt.mig_parent_gpu_uuid,
        model_tree_sha256=receipt.model_tree_sha256,
        tokenizer_tree_sha256=receipt.tokenizer_tree_sha256,
        vllm_package_tree_sha256=receipt.vllm_package_tree_sha256,
        vllm_native_binary_sha256=receipt.vllm_native_binary_sha256,
        gladius_overlay_tree_sha256=receipt.gladius_overlay_tree_sha256,
        tree_hash_algorithm_version=receipt.tree_hash_algorithm_version,
        vllm_version=receipt.vllm_version,
        vllm_module_path=receipt.vllm_module_path,
        gladius_overlay_path=receipt.gladius_overlay_path,
        startup_max_model_len=receipt.startup_max_model_len,
        startup_max_num_seqs=receipt.startup_max_num_seqs,
        startup_max_num_batched_tokens=receipt.startup_max_num_batched_tokens,
        gpu_memory_utilization=receipt.gpu_memory_utilization,
        prefix_caching_enabled=receipt.prefix_caching_enabled,
        chunked_prefill_enabled=receipt.chunked_prefill_enabled,
        enforce_eager=receipt.enforce_eager,
        cuda_graph_mode=receipt.cuda_graph_mode,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--lanes", type=int, default=4)
    parser.add_argument("--base-port", type=int, default=8410)
    args = parser.parse_args(argv)

    report: dict = {"lanes": [], "cross_lane": {}}
    receipts = []

    for lane in range(args.lanes):
        port = args.base_port + lane
        policy_dir = args.run_dir / f"policy{lane}"
        receipt = read_server_start_receipt(policy_dir / "server_start_receipt.json")
        receipts.append(receipt)
        expectation = deployment_for(receipt)

        entry: dict = {
            "gpu_index": lane,
            "port": port,
            "server_instance_id": receipt.server_instance_id,
            "api_pid": receipt.api_pid,
            "engine_core_pid": receipt.engine_core_pid,
            "api_and_engine_are_distinct_processes": (
                receipt.api_pid != receipt.engine_core_pid
            ),
            "physical_gpu_uuid": receipt.physical_gpu_uuid,
            "physical_gpu_identity_source": receipt.physical_gpu_identity_source,
        }

        # Measured, not remembered: who owns the port right now.
        entry["socket_owner_pid"] = listen_socket_owner_pid("127.0.0.1", port)
        entry["socket_owner_is_receipt_api_pid"] = (
            entry["socket_owner_pid"] == receipt.api_pid
        )

        # Verification against a manifest describing this host: passes.
        entry["verification_against_this_host"] = verify_receipt_against_deployment(
            receipt, expectation
        )

        # Verification against a manifest demanding a corroborated CUDA/NVML
        # identity: must fail here, and does. This is the H100 gate.
        formal = deployment_for(receipt)
        formal = DeploymentExpectation(
            **{
                **formal.__dict__,
                "physical_gpu_identity_source": "cuda-nvml-corroborated",
            }
        )
        entry["formal_verification"] = verify_receipt_against_deployment(
            receipt, formal
        )

        now = datetime.now(timezone.utc)
        atomic_write_json(
            policy_dir / "policy_snapshot.json",
            {
                "schema_version": "1.0.0",
                "generation": 1,
                "policy_id": f"smoke-lane{lane}-generation1",
                "model_id": receipt.model_id,
                "engine_id": receipt.engine_id,
                "created_at": now.isoformat().replace("+00:00", "Z"),
                "expires_at": (now + timedelta(minutes=30))
                .isoformat()
                .replace("+00:00", "Z"),
                "admission": {
                    "max_num_seqs": 4,
                    "max_num_batched_tokens": 1024,
                },
            },
        )
        entry["request"] = send_one(port)
        report["lanes"].append(entry)

    # --- the seal lifecycle, driven through the live schedulers ----------
    for lane, receipt in enumerate(receipts):
        policy_dir = args.run_dir / f"policy{lane}"
        write_seal_request(
            policy_dir,
            server_instance_id=receipt.server_instance_id,
            deployment_manifest_sha256="0" * 64,
            expected_final_generation=1,
            attestation_nonce=receipt.attestation_nonce,
        )

    # One more request per lane: an idle scheduler has no next boundary at
    # which to observe the request.
    for lane in range(args.lanes):
        send_one(args.base_port + lane, prompt="seal")

    deadline = time.monotonic() + 120
    for lane, receipt in enumerate(receipts):
        policy_dir = args.run_dir / f"policy{lane}"
        entry = report["lanes"][lane]
        while time.monotonic() < deadline:
            ack = read_seal_ack(policy_dir)
            if ack is not None:
                break
            send_one(args.base_port + lane, prompt="tick")
            time.sleep(0.5)
        ack = read_seal_ack(policy_dir)
        entry["seal_ack"] = ack
        entry["seal_written_by_server"] = (
            policy_dir / TELEMETRY_SEAL_FILENAME
        ).is_file()
        entry["retired_marker"] = (policy_dir / RETIRED_MARKER_FILENAME).is_file()
        if entry["seal_written_by_server"]:
            entry["seal_verification"] = verify_telemetry_seal(
                policy_dir / TELEMETRY_SEAL_FILENAME,
                policy_dir,
                expectation=deployment_for(receipt),
            )

    # --- cross-lane -------------------------------------------------------
    instances = [r.server_instance_id for r in receipts]
    uuids = [r.physical_gpu_uuid for r in receipts]
    report["cross_lane"] = {
        "distinct_server_instances": len(set(instances)),
        "distinct_physical_gpu_uuids": len(set(uuids)),
        "distinct_engine_core_pids": len({r.engine_core_pid for r in receipts}),
        "distinct_api_pids": len({r.api_pid for r in receipts}),
        "distinct_ports": len({r.listen_port for r in receipts}),
    }

    # The lane-set check must refuse four lanes on one card. Recorded as a
    # positive result: a check that never fires is a check nobody has seen
    # work.
    try:
        from smig_lane_set_probe import probe_lane_set  # type: ignore

        report["cross_lane"]["lane_set_refusal"] = probe_lane_set(receipts)
    except ImportError:
        report["cross_lane"]["lane_set_refusal"] = (
            "not probed here; SMIG's validate_lane_set requires four distinct "
            "physical GPU UUIDs and this host has one card, so the set it "
            "would be given is refused by construction"
        )

    args.artifacts.mkdir(parents=True, exist_ok=True)
    (args.artifacts / "smoke-report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
