"""Per-split success rates in the leaderboard format, from ``<log-dir>/evals/<split>/<task>/*/stats.json``.

Same aggregation as robocasa-benchmark/openpi ``examples/robocasa/get_eval_stats.py`` (latest
stats.json per task, mean over the tasks of each TARGET_TASKS group), plus the leaderboard's
task-weighted overall (18 atomic_seen + 16 composite_seen + 16 composite_unseen).

    python sim/summarize.py --log-dir results/<run>
"""

import argparse
import json
from pathlib import Path

from robocasa.utils.dataset_registry import TARGET_TASKS

GROUPS = ("atomic_seen", "composite_seen", "composite_unseen")


def task_rates(log_dir: Path, split: str) -> dict[str, float]:
    out = {}
    split_dir = log_dir / "evals" / split
    for task_dir in sorted(p for p in split_dir.glob("*") if p.is_dir()):
        stamps = sorted(p for p in task_dir.iterdir() if (p / "stats.json").exists())
        if stamps:
            out[task_dir.name] = 100.0 * json.loads((stamps[-1] / "stats.json").read_text())["success_rate"]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-dir", type=Path, required=True)
    ap.add_argument("--split", default="pretrain")
    args = ap.parse_args()
    rates = task_rates(args.log_dir, args.split)
    fields, complete = {}, True
    for group in GROUPS:
        tasks = TARGET_TASKS[group]
        done = [t for t in tasks if t in rates]
        print(f"== {group}: {len(done)}/{len(tasks)} tasks evaluated")
        for t in done:
            print(f"   {t:32s} {rates[t]:5.1f}")
        if done:
            fields[f"{group}_success"] = round(sum(rates[t] for t in done) / len(done), 1)
            print(f"   {'AVG':32s} {fields[f'{group}_success']:5.1f}")
        complete &= len(done) == len(tasks)
    if complete:
        overall = (18 * fields["atomic_seen_success"] + 16 * fields["composite_seen_success"]
                   + 16 * fields["composite_unseen_success"]) / 50
        print(f"\noverall (task-weighted): {overall:.1f}")
    else:
        print("\n(partial run: averages cover only the evaluated tasks; not a leaderboard number)")
    print("leaderboard fields:", json.dumps(fields))


if __name__ == "__main__":
    main()
