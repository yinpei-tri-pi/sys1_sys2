# sys1_sys2: RoboCasa365 evaluation of a System2 + System1 policy

Evaluation-only package for a hierarchical policy on the
[RoboCasa365 leaderboard](https://github.com/robocasa-benchmark/leaderboard) protocol.

- **System2**, a Qwen3-VL-4B planner served by vLLM in Docker, decides *what* to do. It writes a
  milestone plan, then after every System1 segment watches a clip of it and issues the next subgoal.
- **System1**, a subgoal-conditioned pi0.5 policy (JAX, openpi) behind a websocket, decides *how* to
  move: action chunks plus its own subgoal-progress estimate.
- **Language steering**, a set of hardcoded revisions of System2's output, sits between the two.
- **Simulator**: official RoboCasa 1.0.1 (git submodule). The loop scores rollouts exactly like the
  official evaluation client.

```
 System2 (vLLM) <--HTTP-- sim/official_eval.py --websocket--> System1 (JAX)
 plan, subgoals            RoboCasa gym env (MuJoCo, EGL)      action chunks
```

## Results

| Atomic, seen | Composite, seen | Composite, unseen | Overall |
| --- | --- | --- | --- |
| 80.1 | 51.5 | 28.1 | **54.3** |

50 tasks x 50 trials, 2500 episodes. Per-task numbers and the exact setup: [doc/results.md](doc/results.md).

## Install

Linux with NVIDIA GPUs (tested on A100-80GB and RTX A6000 48 GB), Docker with the NVIDIA container
toolkit, git, and [uv](https://docs.astral.sh/uv/) (`curl -LsSf https://astral.sh/uv/install.sh | sh`).
About 60 GB of disk: 30 GB System2 image, 10 GB kitchen assets, 22 GB checkpoints.

```bash
git clone --recursive <this repo> sys1_sys2 && cd sys1_sys2
bash scripts/setup.sh                 # System1 env, System2 image, simulator env + kitchen assets
bash scripts/download_checkpoints.sh  # both checkpoints from the Hugging Face Hub, see Checkpoints
```

`setup.sh` creates `system1/.venv` (pinned by `system1/uv.lock`), builds the Docker image
`sys1-sys2/system2-vllm:0.19.1` and creates `sim/.venv` with RoboCasa and robosuite from the
submodules.

## Checkpoints

The evaluated checkpoints are public on the Hugging Face Hub; no token is needed.

| | Hugging Face repo | Size | Local path (default for every script) |
| --- | --- | --- | --- |
| System1 | [`yinpei-tri-pi/robocasa-system1-pi05-progact`](https://huggingface.co/yinpei-tri-pi/robocasa-system1-pi05-progact) | 12 GB | `checkpoints/system1/f0717-...-progact__.../269999` |
| System2 | [`yinpei-tri-pi/robocasa-system2-qwen3vl-4b`](https://huggingface.co/yinpei-tri-pi/robocasa-system2-qwen3vl-4b) | 10 GB | `checkpoints/system2/system2-full-0804-...-ep3/checkpoint-17124` |

`bash scripts/download_checkpoints.sh` downloads both into those paths. The exact paths are in
`scripts/checkpoints.env`. Other checkpoints work with `--s1-ckpt` and `--s2-ckpt`. System1 takes a
step dir with `params/`, `assets/robocasa_system1/norm_stats.json` and `config.json`. System2 takes a
merged Hugging Face checkpoint.

## Run

```bash
bash scripts/run_fleet.sh --num-trials 2   # 100-episode check, ~35 min on 8 GPUs
bash scripts/run_fleet.sh                  # full benchmark: 50 tasks x 50 trials on every GPU
```

`run_fleet.sh` starts one complete stack per GPU: System1, System2 and a simulator client. The
stacks take whole tasks from a shared queue, longest horizon first. Each task's trials run in order
on one env, as the official client does. With 80 GB GPUs the default memory split (`--s1-mem-frac
0.32 --s2-mem-frac 0.42`) fits a stack on one card. On smaller cards, give System2 its own GPUs with
`--s2-gpus`.

- `--gpus 0,1,2,3` picks GPUs. `--tasks` takes task names and/or `atomic_seen`,
  `composite_seen`, `composite_unseen`, `all`, comma-separated.
- Results go to `results/<protocol>__<run name>/`, e.g.
  `results/official__s1-progact270k_s2-qwen3vl-4b-full-ep3-17124/`. Set it with `--log-dir`.
- Relaunching the same command resumes: a task with a `stats.json` is skipped. An interrupted task
  restarts from trial 0.
- Monitor with `sim/.venv/bin/python sim/progress.py --log-dir <results dir>`. The fleet writes
  `timing.json` and `summary.txt` when it ends.
- Loop flags go after `--`, e.g. `-- --no-task-lang-steer --no-general-lang-steer`. See
  `sim/.venv/bin/python sim/official_eval.py --help`.
- A cold vLLM start can take several minutes. Docker setups that reject `--gpus` need
  `--docker-gpu-mode runtime`.

### Full 50-task evaluation on several GPUs

The leaderboard run is 50 tasks x 50 trials (2500 episodes). On one machine with 8 GPUs, one stack per
GPU, with every default spelled out:

```bash
tmux new -s fulleval        # or setsid nohup ... &: the run takes about half a day
bash scripts/run_fleet.sh \
  --s1-ckpt checkpoints/system1/f0717-270k-bs512-progact__progact_granfine_verbsimp_noexec/269999 \
  --s2-ckpt checkpoints/system2/system2-full-0804-qwen3vl-4b-gb192-full-vitfull-lr1e5-vitlr2e6-alignerlr1e5-zero2-2n-ep3/checkpoint-17124 \
  --gpus 0,1,2,3,4,5,6,7 --s1-mem-frac 0.32 --s2-mem-frac 0.42 \
  --protocol official --tasks all --num-trials 50 --split pretrain --seed 7 \
  --log-dir results/full_official
```

While it runs, and after it ends:

```bash
sim/.venv/bin/python sim/progress.py --log-dir results/full_official    # episodes done, elapsed, ETA
sim/.venv/bin/python sim/summarize.py --log-dir results/full_official   # per-task and per-split success
cat results/full_official/timing.json                                   # start, end, wall time
```

On GPUs smaller than 80 GB, run System2 on its own cards: `--gpus 0,1,2,3 --s2-gpus 4,5,6,7` gives four
stacks, System1 and the simulator on GPUs 0-3 and System2 on GPUs 4-7. On 8 x A100-80GB the
100-episode pilot (`--num-trials 2`) took 35 minutes, which puts the full run at about 13 hours.

One stack on two GPUs, or the servers on their own:

```bash
bash scripts/run_eval.sh --s1-gpu 0 --s2-gpu 1 --tasks CloseFridge,OpenDrawer --num-trials 2
bash scripts/serve_system1.sh --gpu 0 --port 8060
bash scripts/serve_system2.sh --gpu 1 --port 8100
```

**Target-split protocol.** `--protocol target` instead runs the 30 recorded RoboCasa target-split
episodes per task (1500) that the training repo reported, restoring each episode's scene and initial
state. It needs the target human datasets (`sim/.venv/bin/python -m robocasa.scripts.download_datasets
--split target --source human`) and `--dataset-root <datasets>/v1.0/target`. It is not the leaderboard
number.

## Outputs and viewer

```
results/<protocol>__<run name>/
  summary.txt, timing.json          leaderboard fields per split + overall; wall time
  evals/pretrain/<task>/<time>/     stats.json {"num_episodes", "success_rate"} (official format), episodes.json
  episodes/<run name>/<task>__official_pretrain_seed7__episode_<k>/
      episode.json                  plan, every System2 turn (prompts, response, parsed tags), language steering
      turn<NN>/                     System1 rollout video, System2 input clip, anchor frame, per-step log
  episodes/<run name>/lang_steer_config.json   which language steering ran (a mismatched rerun is refused)
  logs/stack<i>/                    system1.log, system2.log, eval.log
```

```bash
sim/.venv/bin/python viewer/server.py --log-dir results/<dir> --port 8090   # http://127.0.0.1:8090/
```

The overview at `/` shows per-task success and plays each episode turn by turn. The rollout
inspector at `/rollout` shows one System2 turn beside the System1 segment it produced. It covers the
prompt System1 received, the video synced to a step slider, and progress, action and gripper curves.
It also shows the predicted action chunks, why the segment ended, and the clip handed to the next
turn. Both read results live.

## Evaluation protocol

`sim/official_eval.py` reproduces the environment side of robocasa-benchmark/openpi
`examples/robocasa/main.py` (commit 5a6beda); only the policy differs.

| | official client | here |
| --- | --- | --- |
| env | `gym.make(f"robocasa/{task}", split="pretrain", seed=7)`, one env per task | same |
| trials | 50 sequential `env.reset()` calls on that env | same |
| instruction | `obs["annotation.human.task_description"]` | same field |
| step limit | `get_task_horizon(task)` env steps | same, counted over all System1 segments |
| success | `info["success"]` after every step; stop at the first success | same |
| cameras | left/right agentview + eye-in-hand, 256x256 | same renders |

- **Scenes.** Trial 0's scene is fixed by the seed. Later trials also depend on how many steps
  earlier trials took, as in the official client. The global `np.random.seed(7)` is reset before
  every task, so a task's result does not depend on which stack ran it.
- **Task status.** System2's "Current task status" line comes from the same success check, so it
  reads `ongoing` whenever System2 is asked.
- **Early stop.** If System2 declares the task finished before the environment does, the
  last-milestone retry restarts the final milestone; an episode that still ends on System2's call
  (4 of 2500 in the reference run) is scored a failure.

## Language steering

`system2/language_steering/` holds hardcoded revisions of System2's output: subgoal wording, step
estimates, plan edits and recovery turns. Some are keyed on the task name, some on the gripper
width. They run in tiers, all on by default, and every intervention is recorded in `episode.json`.

| Tier | Steers |
| --- | --- |
| mandatory | `repeat_cap`: advance a subgoal issued 3 times (2 for a reach) |
| general | `press_again`, `regrasp_recovery` (one injected re-grasp after a missed grasp), `sink_faucet_est` |
| task | 24 steers for 11 tasks, a plan override for StackBowlsCabinet, a gripper pin for GetToastedBread (`--no-task-lang-steer` turns them off) |
| last-milestone retry | `final_milestone_retry`: retry the last milestone instead of accepting a false finish |

`sim/.venv/bin/python -m pytest tests -q` runs their unit tests.

## Layout

```
system1/                 System1 server: inference-only subset of openpi (uv project), serve_system1.py
system2/                 System2 Dockerfile (vLLM 0.19.1); language_steering/
sim/                     official_eval.py (leaderboard driver), target_eval.py (target-split driver),
                         combined_eval.py (the System2 <-> System1 episode loop), sys2_client.py,
                         stop_criterion.py, summarize.py, progress.py, work_queue.py, vendor/ (frame sampling)
viewer/                  results website (standard library only)
scripts/                 setup.sh, download_checkpoints.sh, run_fleet.sh, run_eval.sh, serve_system{1,2}.sh
doc/results.md           reference results
tests/                   language steering, System2 response parser, work queue
third_party/             official RoboCasa @ 2ce1aec (v1.0.1), robosuite @ 232ce7d4 (v1.5.2)
```

## Provenance

| here | source |
| --- | --- |
| `system1/src/openpi/**`, `system1/packages/openpi-client` | openpi fork `yinpei-tri/openpi` @ e5bad79a; `training/config.py`, `policies/policy_config.py` and `serve_system1.py` are inference-only slims with the same transforms and model config |
| `sim/combined_eval.py`, `sim/sys2_client.py`, `sim/stop_criterion.py`, `system2/language_steering/*` | the same fork's `examples/robocasa/` (renamed; logic unchanged) |
| `sim/s1_io.py` | the 8 symbols the loop uses from `examples/robocasa/subtask_eval.py` |
| `sim/subtask_env.py` | robocasa fork `yinpei-tri/robocasa` @ 26a7ba99 |
| `sim/vendor/sys2/`, `scripts/serve_system2.sh` flags | `yinpei-tri/sys2_train_eval` @ 189c24b3 |
| `viewer/rollout.html` | openpi fork `examples/robocasa/subtask_eval_gui.py`, `/combine` page |

Parity checks on A100 (2026-10-25): the System1 server here and openpi's own server returned the same
prompt and actions within 3e-3, inside the 1.6e-2 spread between two fresh openpi servers. Replaying
24 recorded System2 requests from the training repo gave 19 byte-identical responses and mean text
similarity 0.98, the same spread the server shows against itself.
