"""File-based work queue shared by the evaluation drivers.

A unit of work is claimed by creating ``<root>/<key>/`` (``mkdir`` is atomic, also on NFS), so any
number of workers -- on one host or several sharing the results dir -- can pull from the same list
without a coordinator. A claim records its owner; a claim whose owner process is gone (same host
only: another host's pid cannot be checked) is reclaimed by the next worker that reaches it.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import socket
import time


class Claims:
    def __init__(self, root: Path, worker: str):
        self.root = root
        self.worker = worker
        self.host = socket.gethostname()
        root.mkdir(parents=True, exist_ok=True)

    def take(self, key: str) -> bool:
        d = self.root / key
        try:
            d.mkdir()
        except FileExistsError:
            if not self._stale(d):
                return False
            # Reclaim from a dead worker on this host: rename away, then race for a fresh mkdir.
            try:
                d.rename(d.with_name(f"{d.name}.stale-{os.getpid()}-{time.time_ns()}"))
                d.mkdir()
            except OSError:
                return False
        (d / "owner.json").write_text(json.dumps(
            {"worker": self.worker, "host": self.host, "pid": os.getpid(), "t": time.time()}))
        return True

    def _stale(self, d: Path) -> bool:
        try:
            owner = json.loads((d / "owner.json").read_text())
        except (OSError, ValueError):
            # mkdir'd but owner.json not written yet (or torn): only stale if it is old.
            return time.time() - d.stat().st_mtime > 600
        if owner.get("host") != self.host:
            return False   # cannot check another host's pid; clear claims/ by hand if it died
        try:
            os.kill(int(owner["pid"]), 0)
            return False
        except ProcessLookupError:
            return True
        except PermissionError:
            return False

    def release(self, key: str) -> None:
        d = self.root / key
        for f in d.glob("*"):
            f.unlink(missing_ok=True)
        d.rmdir()
