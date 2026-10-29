"""Default results-dir name for a run: ``<protocol>__<method>``.

    official__s1-progact270k_s2-qwen3vl-4b-full-ep3-17124      leaderboard protocol, 50 trials per task
    official-2trials__s1-...                                     fewer trials (a pilot)
    target30__s1-...                                             30 target-split dataset episodes per task

``<method>`` is the name the episode loop gives the episodes dir (from the two checkpoint dirs), so the
results dir and ``episodes/<method>/`` always agree. Non-default --split/--seed are appended.
"""

import argparse
import contextlib
import os
from pathlib import Path
import sys

_HERE = Path(__file__).resolve().parent
os.environ.setdefault("SYS2_REPO", str(_HERE / "vendor"))
sys.path.insert(0, str(_HERE))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", choices=["official", "target"], required=True)
    ap.add_argument("--s1-dir", required=True)
    ap.add_argument("--s2-dir", required=True)
    ap.add_argument("--num-trials", type=int, default=50)
    ap.add_argument("--split", default="target")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    # robocasa prints warnings to STDOUT on import; keep stdout for the name alone.
    with contextlib.redirect_stdout(sys.stderr):
        import combined_eval as CE  # heavy import (robocasa); only needed for the method name

    if a.protocol == "target":
        label = "target30"
    else:
        label = "official" if a.num_trials == 50 else f"official-{a.num_trials}trials"
        if a.split != "target":
            label += f"-{a.split}"
        if a.seed != 7:
            label += f"-seed{a.seed}"
    print(f"{label}__{CE._short_method_name(a.s1_dir, a.s2_dir)}")


if __name__ == "__main__":
    main()
