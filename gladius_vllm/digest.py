"""Canonical content and process-identity digests for the server-start receipt.

Two problems this module exists to solve:

1. A digest that means "the current working tree" proves nothing to a
   reviewer. `tree_sha256()` therefore enumerates exactly which files it
   covered, in a documented order, including each entry's type and size, so
   the same bytes always produce the same digest and a reviewer can re-derive
   it with the published algorithm version.
2. A PID alone cannot identify a process, because PIDs are reused.
   `process_start_identity()` binds the kernel boot id to the process's
   start time (jiffies since boot), which together are unique for the life of
   the machine.
"""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

TREE_HASH_ALGORITHM_VERSION = "gladius-tree-sha256-v1"

# Transient build/runtime droppings that are not part of the shipped source
# and would otherwise make an identical checkout hash differently depending
# on whether it had been imported yet.
_EXCLUDED_DIR_NAMES = frozenset({"__pycache__", ".git", ".mypy_cache", ".ruff_cache"})
_EXCLUDED_SUFFIXES = (".pyc", ".pyo")

_CHUNK_BYTES = 1024 * 1024


class DigestError(RuntimeError):
    """A digest could not be computed from real, enumerable inputs."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(_CHUNK_BYTES)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _tree_entries(root: Path) -> list[str]:
    entries: list[str] = []
    for directory, dir_names, file_names in os.walk(root, followlinks=False):
        dir_names[:] = sorted(
            name for name in dir_names if name not in _EXCLUDED_DIR_NAMES
        )
        for file_name in sorted(file_names):
            if file_name.endswith(_EXCLUDED_SUFFIXES):
                continue
            absolute = Path(directory) / file_name
            relative = absolute.relative_to(root).as_posix()
            if absolute.is_symlink():
                # Hash the link target text, not the resolved content: a
                # symlink swap is a real change to the tree even when the
                # destination bytes are identical.
                target = os.readlink(absolute)
                entries.append(
                    f"{relative}\0symlink\0{len(target.encode('utf-8'))}\0"
                    f"{sha256_text(target)}"
                )
                continue
            stat_result = absolute.stat()
            entries.append(
                f"{relative}\0file\0{stat_result.st_size}\0{sha256_file(absolute)}"
            )
    return sorted(entries)


def tree_sha256(root: Path) -> str:
    """Digest a directory tree (or a single file) reproducibly.

    Entries are `relpath\\0kind\\0size\\0content_sha256`, sorted by the joined
    entry string, newline-terminated, prefixed by the algorithm version so a
    future algorithm change can never collide with this one.
    """
    root = Path(root)
    if not root.exists():
        raise DigestError(f"cannot digest missing path {root}")
    if root.is_file():
        entries = [f"{root.name}\0file\0{root.stat().st_size}\0{sha256_file(root)}"]
    else:
        entries = _tree_entries(root)
    if not entries:
        raise DigestError(f"cannot digest empty tree {root}")
    body = "".join(f"{entry}\n" for entry in entries)
    return sha256_text(f"{TREE_HASH_ALGORITHM_VERSION}\n{body}")


def loaded_native_extension_manifest(package_root: Path) -> str:
    """Ordered manifest digest of every loaded native extension under `root`.

    `sys.modules` is the authority on what this interpreter actually loaded;
    scanning the directory for `.so` files would also count extensions that
    were never imported.
    """
    package_root = Path(package_root).resolve()
    entries: list[str] = []
    for module in list(sys.modules.values()):
        module_file = getattr(module, "__file__", None)
        if not module_file or not module_file.endswith((".so", ".pyd", ".dylib")):
            continue
        path = Path(module_file).resolve()
        try:
            relative = path.relative_to(package_root).as_posix()
        except ValueError:
            continue
        entries.append(f"{relative}\0{path.stat().st_size}\0{sha256_file(path)}")
    body = "".join(f"{entry}\n" for entry in sorted(set(entries)))
    # An empty manifest is a legitimate observation (a pure-Python install),
    # and is still a *distinct* digest from any non-empty one.
    return sha256_text(f"{TREE_HASH_ALGORITHM_VERSION}\nnative\n{body}")


def boot_identity() -> str:
    """Kernel boot id: distinguishes PIDs reused across a reboot."""
    try:
        return Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    except OSError as exc:
        raise DigestError("cannot read kernel boot id") from exc


def process_start_identity(pid: int) -> str:
    """`<boot_id>:<starttime_jiffies>` for `pid`.

    Reused PIDs get different start times, so this value identifies one
    concrete process for the life of the booted machine.
    """
    try:
        stat_text = Path(f"/proc/{pid}/stat").read_text()
    except OSError as exc:
        raise DigestError(f"cannot read /proc/{pid}/stat") from exc
    # The `comm` field is parenthesised and may itself contain spaces and
    # parentheses, so fields are only unambiguous after the final ')'.
    closing = stat_text.rfind(")")
    if closing == -1:
        raise DigestError(f"malformed /proc/{pid}/stat")
    fields = stat_text[closing + 2 :].split()
    # /proc(5) field 22 (starttime) is index 19 once pid and comm are gone.
    if len(fields) <= 19:
        raise DigestError(f"truncated /proc/{pid}/stat")
    return f"{boot_identity()}:{fields[19]}"


def derive_server_instance_id(
    *,
    attestation_nonce: str,
    api_pid: int,
    api_process_start_identity: str,
    engine_core_pid: int,
    engine_core_process_start_identity: str,
    engine_id: str,
    model_id: str,
    physical_gpu_uuid: str,
) -> str:
    """Bind one API/EngineCore process pair to one unforgeable identifier.

    Every component is a measured fact: the launcher's unpredictable nonce,
    both processes' reuse-proof identities, the served identity, and the GPU
    observed from inside the model process.
    """
    material = "\0".join(
        (
            "gladius-server-instance-v1",
            attestation_nonce,
            str(api_pid),
            api_process_start_identity,
            str(engine_core_pid),
            engine_core_process_start_identity,
            engine_id,
            model_id,
            physical_gpu_uuid,
        )
    )
    return f"srv-{sha256_text(material)[:32]}"
