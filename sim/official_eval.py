"""RoboCasa365 leaderboard evaluation of the System2 + System1 closed loop.

The environment lifecycle is the official one, reproduced from robocasa-benchmark/openpi
``examples/robocasa/main.py`` (commit 5a6beda) line for line:

    np.random.seed(seed)                                         # once, before the task loop
    for env_name in tasks:
        env = gym.make(f"robocasa/{env_name}", split=split, seed=seed)
        for episode_idx in range(num_trials):
            obs, info = env.reset()
            ... step until info["success"] or t == get_task_horizon(env_name) ...
        write <log_dir>/evals/<split>/<env_name>/<timestamp>/stats.json

Only the policy differs: instead of one websocket policy, each episode runs ``combined_eval``'s
System2 (planner, vLLM) -> System1 (pi0.5, websocket) loop on the env that ``env.reset()`` just
produced. That loop already scores exactly like the official client:
  * success is ``env._check_success()`` polled after EVERY env step (``info["success"]`` in the gym
    wrapper is the same call), and the episode ends at the first success;
  * the step budget is the cumulative number of env steps, capped at the registry horizon
    (``--rollout-limit-mode max_official_steps``, the same ``get_task_horizon`` value);
  * the task instruction is ``env.get_ep_meta()["lang"]`` (the gym wrapper's
    ``annotation.human.task_description``).
System2's "Current task status" line is computed from that same success check. Because the episode
ends on the first success, it can only ever read "ongoing" when System2 is queried.

Run in the robocasa env with a System1 server and a System2 server up (see README):

    MUJOCO_GL=egl python sim/official_eval.py --tasks CloseFridge OpenDrawer --num-trials 2 \\
        --s1-port 8060 --s2-port 8100 --s1-dir <s1 ckpt> --s2-dir <s2 ckpt> \\
        --norm-stats <s1 ckpt>/assets/robocasa_system1/norm_stats.json --log-dir results/smoke
"""

from __future__ import annotations

import datetime
import json
import os
from pathlib import Path
import sys
import time

# sys2_client resolves the System2 frame-sampling policy (sys2.data.video_policy) from $SYS2_REPO at
# import time; point it at the vendored copy before combined_eval imports it.
_HERE = Path(__file__).resolve().parent
os.environ.setdefault("SYS2_REPO", str(_HERE / "vendor"))
sys.path.insert(0, str(_HERE))

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
from openpi_client import websocket_client_policy as _wcp  # noqa: E402
import robocasa  # noqa: E402,F401  (registers the robocasa/<task> gym ids)
from robocasa.utils.dataset_registry import TARGET_TASKS  # noqa: E402
from robocasa.utils.dataset_registry_utils import get_task_horizon  # noqa: E402

import combined_eval as CE  # noqa: E402
import sys2_client as S2C  # noqa: E402

GYM_CAMERA_SIZE = 256  # RoboCasaGymEnv's camera config (PandaOmronKeyConverter.get_camera_config)


class _SharedEnv:
    """The gym wrapper's inner robosuite env, shared across a task's trials.

    ``combined_eval.eval_episode`` closes its env when an episode ends; the official protocol keeps
    ONE env per task and calls ``reset()`` on it for every trial, so close() is a no-op here and the
    real close happens once the task is finished.
    """

    def __init__(self, env):
        object.__setattr__(self, "_env", env)

    def __getattr__(self, name):
        return getattr(self._env, name)

    def __setattr__(self, name, value):
        setattr(self._env, name, value)

    def close(self):
        pass


class OfficialResetAdapter:
    """Feeds ``combined_eval.eval_episode`` the official env instead of a recorded LeRobot episode.

    eval_episode builds its env with ``make_camera_env`` and restores a dataset frame with
    ``reset_to``. Both are replaced: the env is the task's ``gym.make`` env after ``env.reset()``,
    and the episode meta (task instruction, layout, objects) is whatever that reset sampled.
    """

    def __init__(self, gym_env, task: str, split: str, seed: int):
        self.gym_env = gym_env
        # gym.make adds gymnasium's checker wrappers; .unwrapped is the RoboCasaGymEnv and its .env the
        # robosuite env that RoboCasaGymEnv.step() itself drives.
        self.robosuite_env = gym_env.unwrapped.env
        self.task = task
        self.split = split
        self.seed = seed

    def install(self) -> None:
        CE.make_camera_env = self.make_env
        CE.reset_to = lambda env, state: None
        CE.LU.get_episode_meta = lambda _ld, _ep: self.robosuite_env.get_ep_meta()
        CE.LU.get_episode_states = lambda _ld, _ep: [None]
        CE.LU.get_episode_model_xml = lambda _ld, _ep: None
        CE._task_name_from_lerobot_dir = lambda _ld: self.task
        CE._split_from_lerobot_dir = lambda _ld: f"official_{self.split}_seed{self.seed}"

    def make_env(self, _ld, camera_h: int = GYM_CAMERA_SIZE, camera_w: int = GYM_CAMERA_SIZE):
        if (camera_h, camera_w) != (GYM_CAMERA_SIZE, GYM_CAMERA_SIZE):
            raise ValueError(f"the official gym env renders {GYM_CAMERA_SIZE}x{GYM_CAMERA_SIZE}; "
                             "--highres-video is not supported here")
        self.gym_env.reset()
        return _SharedEnv(self.robosuite_env)


def _expand_tasks(keys: list[str]) -> list[str]:
    """``main.py`` semantics: a TARGET_TASKS group name expands to its tasks, anything else is a task."""
    groups = {"all": [t for g in ("atomic_seen", "composite_seen", "composite_unseen") for t in TARGET_TASKS[g]]}
    groups.update(TARGET_TASKS)
    out: list[str] = []
    for key in keys:
        out.extend(groups.get(key, [key]))
    return out


def _task_done(task_dir: Path) -> bool:
    return task_dir.is_dir() and any(task_dir.glob("*/stats.json"))


def build_argparser():
    ap = CE.build_argparser(
        "RoboCasa365 leaderboard protocol (robocasa-benchmark/openpi main.py) for the S2+S1 loop")
    # Every combined_eval loop flag stays available; only the episode source changes.
    for action in ap._actions:
        if action.dest == "lerobot_dir":
            action.required = False
            action.default = None
            action.help = "unused: episodes come from the official gym env"
    ap.set_defaults(rollout_limit_mode="max_official_steps", stop_on_env_success=True)
    off = ap.add_argument_group("official protocol")
    off.add_argument("--tasks", nargs="+", default=["atomic_seen"],
                     help="task names and/or TARGET_TASKS groups (atomic_seen composite_seen "
                          "composite_unseen, or all); run sequentially in ONE process like main.py")
    off.add_argument("--split", choices=["pretrain", "target"], default="pretrain",
                     help="scene split (the leaderboard evaluates in pretrain scenes)")
    off.add_argument("--seed", type=int, default=7)
    off.add_argument("--num-trials", type=int, default=50, help="rollouts per task (leaderboard: 50)")
    off.add_argument("--log-dir", type=Path, required=True,
                     help="results root: evals/<split>/<task>/<timestamp>/stats.json + episodes/")
    return ap


def main() -> None:
    args = build_argparser().parse_args()
    if args.rollout_limit_mode != "max_official_steps":
        raise SystemExit("the leaderboard protocol caps each episode at the official horizon; "
                         "keep --rollout-limit-mode max_official_steps")
    if args.max_official_steps is not None or args.task_limits_json:
        raise SystemExit("--max-official-steps/--task-limits-json would replace the official horizon")
    if args.goal_json:
        raise SystemExit("--goal-json would replace the environment's task instruction")
    # Same tier resolution as combined_eval.run_sweep (task rules imply general rules).
    args.general_rules = bool(args.general_rules or args.task_rules)
    if args.last_milestone_retry is None:
        args.last_milestone_retry = args.general_rules

    tasks = _expand_tasks(args.tasks)
    log_dir = args.log_dir.resolve()
    out_root = log_dir / "episodes"
    if not args.method:
        args.method = CE._short_method_name(args.s1_dir, args.s2_dir)
    CE._write_run_rule_config(out_root, args.method, args)
    norm_stats = None
    if args.norm_stats and args.norm_stats.exists():
        raw = json.loads(args.norm_stats.read_text())
        norm_stats = raw.get("norm_stats", raw)

    s1 = _wcp.WebsocketClientPolicy(host=args.s1_host, port=args.s1_port)
    s2 = S2C.Sys2Client(args.s2_host, args.s2_port, args.s2_model,
                        max_tokens=args.s2_max_tokens, inline_media=not args.s2_file_uri)
    print(f"System2 server health: {s2.health()}  (model={args.s2_model})", flush=True)

    np.random.seed(args.seed)
    summary = {}
    for task in tasks:
        task_dir = log_dir / "evals" / args.split / task
        if _task_done(task_dir):
            print(f"{task}/{args.split}: stats.json exists, skipping.", flush=True)
            continue
        horizon = get_task_horizon(task)
        print(f"\n##### {task}  split={args.split} seed={args.seed} trials={args.num_trials} "
              f"horizon={horizon}", flush=True)
        t_task = time.time()
        gym_env = gym.make(f"robocasa/{task}", split=args.split, seed=args.seed)
        adapter = OfficialResetAdapter(gym_env, task, args.split, args.seed)
        adapter.install()
        results = []
        try:
            for trial in range(args.num_trials):
                print(f"=== {task} trial {trial + 1}/{args.num_trials} ===", flush=True)
                args.episode_index = trial
                doc = CE.eval_episode(Path(f"official/{task}"), s1, s2, args, out_root, args.method,
                                      norm_stats)
                if doc.get("error"):
                    # An infrastructure failure (server down, crash) is not a policy failure; abort
                    # the task rather than score it, so a rerun starts the task from trial 0.
                    raise RuntimeError(f"{task} trial {trial}: {doc['error']}")
                if doc.get("max_official_steps") != horizon:
                    raise RuntimeError(f"{task}: loop horizon {doc.get('max_official_steps')} != "
                                       f"official horizon {horizon}")
                doc["lerobot_dir"] = None
                doc["config"]["reset_mode"] = (f'official: gym.make("robocasa/{task}", split='
                                               f'"{args.split}", seed={args.seed}) then env.reset() '
                                               f"for trial {trial}")
                doc["official_protocol"] = {"split": args.split, "seed": args.seed, "trial": trial,
                                            "horizon": horizon, "num_trials": args.num_trials}
                CE._write_json(out_root / args.method / doc["episode_id"].replace("/", "__")
                               / "episode.json", doc)
                results.append(doc)
                n_ok = sum(bool(r.get("episode_success")) for r in results)
                print(f"  success={bool(doc.get('episode_success'))} termination={doc.get('termination')} "
                      f"steps={doc.get('n_env_steps')}  [{task}: {n_ok}/{len(results)}]", flush=True)
        finally:
            gym_env.close()

        n_ok = sum(bool(r.get("episode_success")) for r in results)
        stats = {"num_episodes": len(results), "success_rate": n_ok / len(results)}
        stamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
        CE._write_json(task_dir / stamp / "stats.json", stats)
        CE._write_json(task_dir / stamp / "episodes.json", {
            "method": args.method, "split": args.split, "seed": args.seed, "horizon": horizon,
            "seconds": round(time.time() - t_task, 1),
            "episodes": [{k: r.get(k) for k in ("episode_id", "episode_success", "termination",
                                                "n_env_steps", "n_turns", "seconds")}
                         for r in results],
        })
        summary[task] = stats["success_rate"]
        print(f"[{task}] success rate {stats['success_rate']:.3f} ({n_ok}/{len(results)})", flush=True)
    print("\nSUMMARY", json.dumps(summary, indent=1), flush=True)


if __name__ == "__main__":
    main()
