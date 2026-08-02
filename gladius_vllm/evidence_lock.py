"""One Linux file lock covering a policy directory's whole evidence lifecycle.

Retirement was previously checked only in the telemetry writer's constructor,
so a handle opened *before* a seal kept appending to a directory that had
already been certified as complete. The certified digests then described
bytes that no longer existed.

The fix has to be a real interprocess lock rather than a Python-level flag:
the appending writer and the sealing path can live in different processes
(the seal is requested from outside the server), so nothing in one process's
memory can order them.
"""

from __future__ import annotations

import contextlib
import fcntl
import os
from collections.abc import Iterator
from pathlib import Path

LOCK_FILENAME = ".gladius-evidence.lock"


@contextlib.contextmanager
def evidence_lock(policy_dir: Path, *, exclusive: bool = True) -> Iterator[None]:
    """Hold the directory's evidence lock for the duration of the block.

    Shared for appends (many writers may hold it, and each rechecks
    retirement while holding it), exclusive for sealing and retirement so
    the transition cannot interleave with an in-flight append.
    """
    policy_dir = Path(policy_dir)
    policy_dir.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(policy_dir / LOCK_FILENAME, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
        try:
            yield
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
    finally:
        os.close(descriptor)
