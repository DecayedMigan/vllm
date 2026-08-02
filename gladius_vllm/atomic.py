"""Crash-safe atomic publication of small JSON control-plane documents.

Every GLADIUS evidence file (`server_start_receipt.json`,
`policy_application.json`, `telemetry_seal.json`) is read by a *separate*
process while the server keeps running, so a partially-written file must
never be observable. The sequence below is the only publication path in this
package: same-directory temporary file -> file `fsync` -> `os.replace` ->
parent-directory `fsync`.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)


def serialize_canonical_json(payload: object) -> str:
    """Deterministic JSON text: sorted keys, no insignificant whitespace.

    Both repositories hash these documents, so the byte encoding has to be a
    pure function of the payload rather than of dict insertion order.
    """
    return json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"


def atomic_write_text(path: Path, text: str) -> None:
    """Publish `text` at `path` atomically. Raises on any failure."""
    temporary_path: Path | None = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary_path is not None:
            with contextlib.suppress(OSError):
                temporary_path.unlink(missing_ok=True)


def atomic_write_json(path: Path, payload: object) -> None:
    """Publish `payload` as canonical JSON at `path`. Raises on any failure."""
    atomic_write_text(path, serialize_canonical_json(payload))


def try_atomic_write_json(path: Path, payload: object, *, what: str) -> bool:
    """Fail-open wrapper: never propagates an I/O error into serving."""
    try:
        atomic_write_json(path, payload)
    except Exception:
        logger.warning("GLADIUS %s write failed for %s", what, path, exc_info=True)
        return False
    return True
