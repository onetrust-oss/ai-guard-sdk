from __future__ import annotations

import fcntl
import socket
import time
from pathlib import Path

_LOCK_FILE = "ports.txt"
_MAX_PORTS = 256


def allocate_port(ports_path: Path) -> int:
    ports_file_path = ports_path / _LOCK_FILE
    ports_file_path.touch(exist_ok=True)

    with open(ports_file_path, "r+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            contents = f.read()
            assigned: list[int] = [
                int(line.strip())
                for line in contents.splitlines()
                if line.strip().isdigit()
            ]

            while True:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.bind(("127.0.0.1", 0))
                port = sock.getsockname()[1]
                sock.close()

                if port not in assigned:
                    assigned.append(port)
                    if len(assigned) > _MAX_PORTS:
                        assigned = assigned[-_MAX_PORTS:]

                    f.seek(0)
                    f.truncate()
                    f.write("\n".join(str(p) for p in assigned))
                    f.flush()

                    time.sleep(0.1)
                    return port

                time.sleep(0.01)
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)
