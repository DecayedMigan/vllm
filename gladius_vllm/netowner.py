"""Who owns a listening socket, read from `/proc` rather than asserted.

Extracted from `attest.py` so the receipt verifier can re-run the same check
immediately before traffic. That matters because a receipt records which PID
owned the endpoint *at attestation time*: by the time a campaign starts
sending, the recorded PID can still be alive and still have its original
start identity while a completely different process has taken the port.
"""

from __future__ import annotations

import socket
from pathlib import Path


class SocketOwnerError(RuntimeError):
    """The listening socket could not be attributed to a visible process."""


def listen_socket_owner_pid(host: str, port: int) -> int:
    """The PID holding the LISTEN socket for `host:port`."""
    inodes = listening_inodes(host, port)
    if not inodes:
        raise SocketOwnerError(f"no process is listening on {host}:{port}")
    for pid_dir in Path("/proc").iterdir():
        if not pid_dir.name.isdigit():
            continue
        try:
            descriptors = list((pid_dir / "fd").iterdir())
        except OSError:
            continue
        for descriptor in descriptors:
            try:
                target = descriptor.readlink().name
            except OSError:
                continue
            if target.startswith("socket:[") and target[8:-1] in inodes:
                return int(pid_dir.name)
    raise SocketOwnerError(
        f"the socket listening on {host}:{port} is not owned by a visible "
        "process; run the attestor as the same user as the API server"
    )


def listening_inodes(host: str, port: int) -> set[str]:
    """Socket inodes in LISTEN state bound to `port` on a matching address."""
    wanted = {hex_address(host, port), hex_address("0.0.0.0", port)}
    if ":" not in host:
        wanted.add(hex_address("::", port, ipv6=True))
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
                ipv6 and local.endswith(f":{port:04X}") and is_any_address(local)
            ):
                inodes.add(fields[9])
    return inodes


def hex_address(host: str, port: int, *, ipv6: bool = False) -> str:
    if ipv6:
        packed = socket.inet_pton(socket.AF_INET6, host)
        words = [
            packed[index : index + 4][::-1].hex().upper() for index in (0, 4, 8, 12)
        ]
        return f"{''.join(words)}:{port:04X}"
    packed = socket.inet_pton(socket.AF_INET, host)
    return f"{packed[::-1].hex().upper()}:{port:04X}"


def is_any_address(local: str) -> bool:
    return set(local.split(":")[0]) == {"0"}
