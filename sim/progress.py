"""Progress of a running fleet: elapsed wall time, episodes and tasks done, rate, ETA.

    python sim/progress.py --log-dir results/<run> [--total 2500]

Elapsed time is measured from the "[fleet] start" line of <log-dir>/fleet.log; when the fleet has
finished, <log-dir>/timing.json holds the final wall time.
"""

import argparse
import datetime as dt
import json
from pathlib import Path
import re


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-dir", type=Path, required=True)
    ap.add_argument("--total", type=int, default=2500, help="episodes in the full run (official: 50 x 50)")
    a = ap.parse_args()
    timing = a.log_dir / "timing.json"
    if timing.exists():
        t = json.loads(timing.read_text())
        print(f"finished: {t['start']} -> {t['end']}, wall time {t['hours']:.2f} h on {t['host']}")
        return
    m = re.search(r"\[fleet\] start (\S+)", (a.log_dir / "fleet.log").read_text())
    if not m:
        raise SystemExit("fleet.log has no '[fleet] start' line (run started before timing was logged)")
    start = dt.datetime.fromisoformat(m.group(1))
    now = dt.datetime.now(start.tzinfo)
    hours = (now - start).total_seconds() / 3600
    def finished(p: Path) -> bool:
        try:
            d = json.loads(p.read_text())
        except (OSError, ValueError):
            return False
        return bool(d.get("termination")) and not d.get("error")

    eps = [p for p in (a.log_dir / "episodes").glob("*/*/episode.json") if finished(p)]
    tasks = sum(1 for d in (a.log_dir / "evals").glob("*/*") if any(d.glob("*/stats.json")))
    running = sorted(p.name for p in (a.log_dir / "claims").glob("*") if p.is_dir()) if (a.log_dir / "claims").exists() else []
    print(f"started {start:%Y-%m-%d %H:%M}, elapsed {hours:.2f} h")
    print(f"episodes {len(eps)}/{a.total}, tasks finished {tasks}, running: {' '.join(running) or '-'}")
    if eps and hours > 0.1:
        # Weight each episode by its task's horizon: the queue runs the longest tasks first, so a
        # plain episodes/hour rate badly overstates the remaining time early in a run.
        try:
            from robocasa.utils.dataset_registry import TARGET_TASKS
            from robocasa.utils.dataset_registry_utils import get_task_horizon
            per_task = a.total // 50
            total_w = per_task * sum(get_task_horizon(t) for g in TARGET_TASKS.values() for t in g)
            done_w = sum(json.loads(p.read_text()).get("max_official_steps") or 0 for p in eps)
        except Exception:  # noqa: BLE001 - fall back to the unweighted estimate
            total_w, done_w = a.total, len(eps)
        frac = done_w / total_w
        left = hours * (1 - frac) / frac
        print(f"{100 * frac:.1f}% of the horizon-weighted work done; ETA {left:.1f} h more "
              f"(~{now + dt.timedelta(hours=left):%a %H:%M}), total ~{hours + left:.1f} h")


if __name__ == "__main__":
    main()
