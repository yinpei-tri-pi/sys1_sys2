"""Target-split dataset protocol: 30 recorded RoboCasa target episodes per task (1500 in total).

This reproduces how the System2 + System1 runs were scored in the training repo (openpi
``examples/robocasa/run_combine_fleet.sh`` with ``USE_EVAL_SET=1``), e.g. the 53.7% run
``s1-progact270k_s2-qwen3vl-4b-full-ep3-17124-inst-retry``:

  * episodes: ``combined_eval.TARGET_EVAL_EPISODES``, the first 30 extracted episodes of every task in
    the RoboCasa365 *target* LeRobot dataset;
  * reset: each episode restores that dataset episode's model XML and initial simulator state
    (``combined_eval.eval_episode``); no recorded action is ever used;
  * scoring: ``env._check_success()`` after every step, stop at the first success, the episode capped
    at the registry horizon (``--rollout-limit-mode max_official_steps``), all language steering on;
  * one fresh ``combined_eval`` process per episode, with the same flags the old fleet passed.

It is NOT the leaderboard protocol (that is ``sim/official_eval.py``: pretrain scenes, seed 7, 50
fresh ``env.reset()`` trials per task). Both share the episode loop.

Work is a shared queue of (task, episode) units under ``<log-dir>/claims/``: any number of workers,
started at any time and on any GPUs, pull from it, so stacks can be added while a run is going. A
finished episode is never redone; a crashed worker's claim is reclaimed by the next worker on the
same host. Units are ordered episode-major (every task's 1st episode, then every task's 2nd, ...),
so a partial run already covers all 50 tasks evenly.

    MUJOCO_GL=egl sim/.venv/bin/python sim/target_eval.py --log-dir results/target30 \\
        --dataset-root $ROBOCASA_LEROBOT_ROOT/v1.0/target --s1-port 8060 --s2-port 8100 \\
        --s1-dir <s1 ckpt> --s2-dir <s2 ckpt> --norm-stats <s1 ckpt>/assets/robocasa_system1/norm_stats.json
    sim/.venv/bin/python sim/summarize.py --log-dir results/target30 --protocol target
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

_HERE = Path(__file__).resolve().parent
os.environ.setdefault("SYS2_REPO", str(_HERE / "vendor"))
sys.path.insert(0, str(_HERE))

import combined_eval as CE  # noqa: E402
from work_queue import Claims  # noqa: E402
from official_eval import _expand_tasks  # noqa: E402

MAX_CONSECUTIVE_FAILURES = 3   # a dead server fails every unit fast; stop instead of draining the queue


def install_replay_compat() -> None:
    """Make official RoboCasa 1.0.1 replay every target task, as the training repo's robocasa fork did.

    ``RecycleBottlesByType`` samples ``self.choice`` in ``_get_obj_cfgs``. A dataset replay takes the
    object configs from the episode meta and never calls that method, so official 1.0.1 raises
    ``AttributeError: ... no attribute 'choice'`` in ``_check_success`` on every replayed episode. The
    fork (yinpei-tri/robocasa @ 26a7ba99) recovers it from the spawned mystery bottle's
    ``obj_groups``; this is that fix, applied at runtime instead of editing the submodule. Fresh
    resets (the leaderboard protocol) set ``choice`` normally and behave exactly as before.
    """
    from robocasa.environments.kitchen.composite.organizing_recycling.recycle_bottles_by_type import (
        RecycleBottlesByType,
    )

    def _get(self):
        if self.__dict__.get("_choice") is not None:
            return self.__dict__["_choice"]
        for cfg in getattr(self, "object_cfgs", None) or []:
            if cfg.get("name") == "mystery_middle":
                groups = cfg.get("obj_groups")
                if isinstance(groups, (list, tuple)):
                    groups = groups[0] if groups else None
                if groups in ("alcohol", "bottled_water"):
                    self.__dict__["_choice"] = groups
                    return groups
        return "alcohol"

    def _set(self, value):
        self.__dict__["_choice"] = value

    RecycleBottlesByType.choice = property(_get, _set)


def lerobot_dir(root: Path, task: str) -> Path | None:
    hits = sorted(root.glob(f"*/{task}/*/lerobot"))
    return hits[0] if hits else None


def build_argparser():
    ap = CE.build_argparser("Target-split dataset protocol (30 episodes per task) for the S2+S1 loop")
    for action in ap._actions:
        if action.dest == "lerobot_dir":
            action.required = False
            action.default = None
            action.help = "set per unit by the worker"
        if action.dest == "out_root":
            action.help = "unused: results go to <log-dir>/episodes"
    ap.set_defaults(rollout_limit_mode="max_official_steps", stop_on_env_success=True)
    g = ap.add_argument_group("target-split protocol")
    g.add_argument("--tasks", nargs="+", default=["all"],
                   help="task names and/or groups: atomic_seen composite_seen composite_unseen all")
    g.add_argument("--log-dir", type=Path, required=True,
                   help="results root: episodes/<method>/<Task>__target__episode_<k>/ + claims/")
    g.add_argument("--dataset-root", type=Path,
                   default=Path(os.environ.get("ROBOCASA_LEROBOT_ROOT", "~/robocasa_dataset")).expanduser()
                   / "v1.0" / "target",
                   help="RoboCasa365 target LeRobot root (<root>/<atomic|composite>/<Task>/<date>/lerobot)")
    g.add_argument("--episodes-per-task", type=int, default=30,
                   help="first N entries of each task's manifest (30 = the reference protocol)")
    g.add_argument("--worker-name", default=f"{socket.gethostname()}-{os.getpid()}")
    g.add_argument("--unit-episode", type=int, default=None, help=argparse.SUPPRESS)  # internal: child mode
    return ap


def run_one_unit(args) -> None:
    """Child process: exactly the old fleet's per-episode ``combined_eval.py --resume`` call."""
    install_replay_compat()
    args.episodes = str(args.unit_episode)
    args.out_root = args.log_dir.resolve() / "episodes"
    args.resume = True
    CE.run_sweep(args)


def main() -> None:
    ap = build_argparser()
    args = ap.parse_args()
    if args.unit_episode is not None:
        return run_one_unit(args)
    if args.rollout_limit_mode != "max_official_steps":
        raise SystemExit("the reference protocol caps each episode at the registry horizon; "
                         "keep --rollout-limit-mode max_official_steps")

    tasks = _expand_tasks(args.tasks)
    unknown = [t for t in tasks if t not in CE.TARGET_EVAL_EPISODES]
    if unknown:
        raise SystemExit(f"not in the 50-task target manifest: {unknown}")
    root = args.dataset_root.expanduser().resolve()
    dirs = {t: lerobot_dir(root, t) for t in tasks}
    missing = [t for t, d in dirs.items() if d is None]
    if missing:
        raise SystemExit(f"no LeRobot dataset under {root} for: {missing} "
                         "(download the target split, see README)")
    n = args.episodes_per_task
    units = [(t, CE.TARGET_EVAL_EPISODES[t][k]) for k in range(n) for t in tasks
             if k < len(CE.TARGET_EVAL_EPISODES[t])]

    log_dir = args.log_dir.resolve()
    if not args.method:
        args.method = CE._short_method_name(args.s1_dir, args.s2_dir)
    method_dir = log_dir / "episodes" / args.method
    # Write/verify the run config once up front, so a mismatched arm fails before any episode runs.
    args.general_lang_steer = bool(args.general_lang_steer or args.task_lang_steer)
    if args.last_milestone_retry is None:
        args.last_milestone_retry = args.general_lang_steer
    CE._write_run_lang_steer_config(log_dir / "episodes", args.method, args)
    claims = Claims(log_dir / "claims", args.worker_name)

    # The child gets this worker's exact argv plus the unit, and an explicit --method so every
    # worker writes into the same method dir even if they were started with different defaults.
    base = [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], "--method", args.method]
    done_before = sum(CE._episode_done(method_dir, dirs[t], e) for t, e in units)
    print(f"[target_eval] {args.worker_name}: {len(units)} units over {len(tasks)} tasks, "
          f"{done_before} already done; method={args.method}", flush=True)
    ran = fails = 0
    for task, ep in units:
        if CE._episode_done(method_dir, dirs[task], ep) or not claims.take(f"{task}__{ep:06d}"):
            continue
        if CE._episode_done(method_dir, dirs[task], ep):   # finished between the check and the claim
            claims.release(f"{task}__{ep:06d}")
            continue
        t0 = time.time()
        print(f"=== {task} episode {ep} ({args.worker_name}) ===", flush=True)
        rc = subprocess.call([*base, "--lerobot-dir", str(dirs[task]), "--unit-episode", str(ep)])
        ok = CE._episode_done(method_dir, dirs[task], ep)
        claims.release(f"{task}__{ep:06d}")
        ran += 1
        if ok:
            fails = 0
            doc = json.loads((method_dir / f"{task}__target__episode_{ep:06d}" / "episode.json").read_text())
            print(f"[target_eval] {task} ep{ep}: success={bool(doc.get('episode_success'))} "
                  f"termination={doc.get('termination')} steps={doc.get('n_env_steps')} "
                  f"({time.time() - t0:.0f}s)", flush=True)
        else:
            fails += 1
            print(f"[target_eval] {task} ep{ep}: NOT completed (exit {rc}); left in the queue", flush=True)
            if fails >= MAX_CONSECUTIVE_FAILURES:
                raise SystemExit(f"{fails} consecutive units failed; check the servers "
                                 "(the queue keeps the remaining units for a relaunch)")
    left = sum(not CE._episode_done(method_dir, dirs[t], e) for t, e in units)
    print(f"[target_eval] {args.worker_name}: ran {ran} units; {left} of {len(units)} not done "
          "(other workers may still hold them)", flush=True)


if __name__ == "__main__":
    main()
