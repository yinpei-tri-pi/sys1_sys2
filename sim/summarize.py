"""Per-split success rates for a results dir.

``--protocol official`` (default): leaderboard format, from ``<log-dir>/evals/<split>/<task>/*/stats.json``.
Same aggregation as robocasa-benchmark/openpi ``examples/robocasa/get_eval_stats.py`` (latest
stats.json per task, mean over the tasks of each TARGET_TASKS group), plus the leaderboard's
task-weighted overall (18 atomic_seen + 16 composite_seen + 16 composite_unseen).

``--protocol target``: the 30-episodes-per-task target-split runs (``sim/target_eval.py``), from
``<log-dir>/episodes/<method>/<Task>__target__episode_<k>/episode.json``. Every task has the same
number of episodes, so the episode-weighted overall equals the task-weighted one (this is how the
training repo's combine_results reported, e.g. 805/1500 = 53.7%).

    python sim/summarize.py --log-dir results/<run> [--protocol target]
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


def official(args) -> None:
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


def target(args) -> None:
    per_task: dict[str, list[bool]] = {}
    errors = 0
    for f in sorted((args.log_dir / "episodes").glob("*/*__target__episode_*/episode.json")):
        try:
            d = json.loads(f.read_text())
        except ValueError:
            continue
        if d.get("error") or not d.get("termination"):
            errors += 1
            continue
        per_task.setdefault(d["task_name"], []).append(bool(d.get("episode_success")))
    tot_n = tot_s = 0
    out = {}
    for group in GROUPS:
        tasks = [t for t in TARGET_TASKS[group] if t in per_task]
        n = sum(len(per_task[t]) for t in tasks)
        s = sum(sum(per_task[t]) for t in tasks)
        print(f"== {group}: {len(tasks)}/{len(TARGET_TASKS[group])} tasks, {n} episodes")
        for t in tasks:
            v = per_task[t]
            print(f"   {t:32s} {100 * sum(v) / len(v):5.1f}  ({sum(v)}/{len(v)})")
        if n:
            out[f"{group}_success"] = round(100 * s / n, 1)
            print(f"   {'ALL':32s} {100 * s / n:5.1f}  ({s}/{n})")
        tot_n += n
        tot_s += s
    if tot_n:
        print(f"\noverall: {100 * tot_s / tot_n:.1f}  ({tot_s}/{tot_n} episodes"
              f"{'' if tot_n == 1500 else ', partial'}){f'; {errors} errored' if errors else ''}")
    print("fields:", json.dumps(out))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-dir", type=Path, required=True)
    ap.add_argument("--protocol", choices=["official", "target"], default="official")
    ap.add_argument("--split", default="pretrain", help="official protocol only")
    args = ap.parse_args()
    (official if args.protocol == "official" else target)(args)


if __name__ == "__main__":
    main()
