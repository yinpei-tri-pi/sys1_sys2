"""The shared work queue: exclusive claims, release, and reclaiming a dead worker's claim."""

import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sim"))
import work_queue as WQ


def test_claim_is_exclusive_and_released(tmp_path):
    a, b = WQ.Claims(tmp_path, "a"), WQ.Claims(tmp_path, "b")
    assert a.take("CloseFridge")
    assert not b.take("CloseFridge")          # live owner (this process)
    a.release("CloseFridge")
    assert b.take("CloseFridge")


def test_dead_owner_on_same_host_is_reclaimed(tmp_path):
    dead = subprocess.Popen([sys.executable, "-c", "pass"]); dead.wait()
    (tmp_path / "OpenDrawer").mkdir()
    (tmp_path / "OpenDrawer" / "owner.json").write_text(json.dumps(
        {"worker": "x", "host": WQ.socket.gethostname(), "pid": dead.pid, "t": 0}))
    assert WQ.Claims(tmp_path, "b").take("OpenDrawer")


def test_other_host_claim_is_never_stolen(tmp_path):
    (tmp_path / "OpenDrawer").mkdir()
    (tmp_path / "OpenDrawer" / "owner.json").write_text(json.dumps(
        {"worker": "x", "host": "some-other-host", "pid": 1, "t": 0}))
    assert not WQ.Claims(tmp_path, "b").take("OpenDrawer")
