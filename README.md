# sys1_sys2 — RoboCasa365 evaluation of the System2 + System1 hierarchy

Evaluation-only package for a dual-system policy on the
[RoboCasa365 leaderboard](https://github.com/robocasa-benchmark/leaderboard) protocol:

- **System2** — a Qwen-VL planner served by **vLLM** (Docker). It decides *what* to do: it writes
  a milestone plan, then after every segment watches a clip of what happened and issues the next
  subgoal.
- **System1** — a subgoal-conditioned **pi0.5** policy (JAX, openpi) served over a websocket. It
  decides *how* to move: it gets the task, the current subgoal and the cameras, and returns action
  chunks plus its own subgoal-progress estimate.
- **Simulator** — **official RoboCasa 1.0.1** (git submodule). The episode loop drives both servers
  and scores rollouts exactly like the official evaluation client.

```
                      HTTP (OpenAI API)                     websocket (openpi)
   System2 (vLLM, GPU) <──────────────── sim/official_eval.py ─────────────────> System1 (JAX, GPU)
   plan / next subgoal                    RoboCasa gym env                        action chunks
                                          (MuJoCo, EGL render)
```

## Layout

```
system1/                 System1 server: inference-only subset of openpi (uv project)
  serve_system1.py       serve one checkpoint
  src/openpi/            model + RoboCasa policy transforms (verbatim from openpi), slim config
system2/
  Dockerfile             System2 server image (vLLM 0.19.1)
  language_steering/     System2 language steering (sys2_steering*.py): general + per-task revisions
                         of System2's output, applied before System1 runs
sim/
  official_eval.py       leaderboard-protocol driver (gym.make / reset / horizon / success)
  combined_eval.py       the S2<->S1 episode loop (verbatim, imports rewritten)
  sys2_client.py         System2 prompts, clip building, response parsing
  stop_criterion.py      System1 segment stop rule (progress + action quiescence)
  s1_io.py, subtask_env.py   System1 request/response + observation/action helpers
  summarize.py           per-split success rates in leaderboard format
  vendor/sys2/           System2 clip frame-sampling policy (must match training)
viewer/                  results website (server.py + index.html, standard library only)
scripts/                 setup + serve + run wrappers
third_party/robocasa     official RoboCasa @ 2ce1aec (v1.0.1)
third_party/robosuite    robosuite @ 232ce7d4 (v1.5.2)
```

## Requirements

- Linux, NVIDIA GPU(s) with the NVIDIA driver; tested on RTX A6000 48 GB.
- Docker with the NVIDIA container toolkit (System2), [uv](https://docs.astral.sh/uv/) (System1 +
  simulator), git.
- Disk: ~25 GB RoboCasa kitchen assets, ~46 GB System2 image, checkpoints (System1 ~12 GB, System2
  ~9 GB).
- GPU memory per stack (measured): System2 at `--mem-frac 0.45` ≈ 21 GB; System1 at
  `--mem-frac 0.35` + MuJoCo EGL rendering ≈ 19 GB. One stack fits on one 48 GB GPU with
  `--s1-mem-frac 0.32 --s2-mem-frac 0.42`, or spread it over two GPUs.

## Install

```bash
git clone --recursive <this repo> sys1_sys2 && cd sys1_sys2
# (already cloned without --recursive: git submodule update --init)

# uv, if missing
curl -LsSf https://astral.sh/uv/install.sh | sh

# 1. System1 env (pinned by system1/uv.lock; same package versions as the training repo)
(cd system1 && uv sync --frozen)

# 2. System2 image (vLLM 0.19.1 on the public ModelScope CUDA 12.9 base)
docker build -t sys1-sys2/system2-vllm:0.19.1 system2

# 3. Simulator env: official robocasa + robosuite from the submodules, macros, ~10 GB assets download
bash scripts/setup_sim_env.sh
```

The System1 tokenizer is downloaded once from `gs://big_vision/paligemma_tokenizer.model`
(public) into `~/.cache/openpi` (override with `OPENPI_DATA_HOME`).

## Checkpoints

| | pass | must contain |
| --- | --- | --- |
| System1 | the **step dir**, e.g. `<run>/269999` | `params/`, `assets/robocasa_system1/norm_stats.json`, and `<run>/config.json` (selects the model/prompt variant, e.g. progress-as-action vs. progress head) |
| System2 | the **checkpoint dir**, e.g. `<run>/checkpoint-17124` | a full (merged) HF checkpoint: `model*.safetensors`, `config.json`, tokenizer/processor files, `chat_template.jinja` |

## Run

One command starts both servers, waits until they are ready, evaluates, stops the servers and
prints the summary:

```bash
bash scripts/run_eval.sh \
  --s1-ckpt /ckpts/sys1/<run>/269999 --s2-ckpt /ckpts/sys2/<run>/checkpoint-17124 \
  --s1-gpu 0 --s2-gpu 1 \
  --tasks atomic_seen --num-trials 50 --log-dir results/atomic_seen
```

- `--tasks` takes task names and/or groups: `atomic_seen` `composite_seen` `composite_unseen`, or
  `all` (the 50 leaderboard tasks), comma-separated (e.g. `atomic_seen,CloseFridge`).
- Quick pipeline check: `--tasks CloseFridge,OpenDrawer,TurnOnSinkFaucet --num-trials 2`.
- A cold vLLM start (image import, torch.compile, CUDA graphs) can take several minutes; System1
  loads in ~1 minute.
- Docker setups that reject `--gpus` (CDI / snap Docker) need `--docker-gpu-mode runtime`.
- Extra loop flags go after `--`, e.g. `-- --no-task-rules --no-general-rules` (see
  `python sim/official_eval.py --help`).

To run the pieces separately (e.g. servers on another host):

```bash
bash scripts/serve_system1.sh --ckpt <s1 step dir> --gpu 0 --port 8060
bash scripts/serve_system2.sh --ckpt <s2 checkpoint dir> --gpu 1 --port 8100
MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=0 sim/.venv/bin/python sim/official_eval.py \
  --tasks atomic_seen --num-trials 50 --log-dir results/atomic_seen \
  --s1-port 8060 --s2-port 8100 --s1-dir <s1 step dir> --s2-dir <s2 checkpoint dir> \
  --norm-stats <s1 step dir>/assets/robocasa_system1/norm_stats.json
sim/.venv/bin/python sim/summarize.py --log-dir results/atomic_seen
```

### Full 50-task evaluation on several GPUs

Each task is one sequential unit: its trials share one env, as in the official client. To
parallelize, run one stack per GPU pair (distinct `--s1-port`/`--s2-port`) on a disjoint subset of
tasks, all writing to the same `--log-dir`; `summarize.py` then reads every task. A task with a
finished `stats.json` is skipped, so an interrupted run can be relaunched with the same command.
An interrupted task restarts from trial 0.

## Outputs

```
<log-dir>/
  evals/<split>/<task>/<timestamp>/stats.json      {"num_episodes", "success_rate"} (official format)
  evals/<split>/<task>/<timestamp>/episodes.json   per-trial outcome, steps, turns
  episodes/<method>/<task>__official_<split>_seed<seed>__episode_<k>/
      episode.json        plan, every System2 turn (prompt, response, parsed tags), language-steering interventions
      turn<NN>/           System1 rollout video, System2 input clip, System1 anchor frame, step log
  episodes/<method>/rule_config.json   which language-steering tiers ran (a mismatched rerun into the same dir is refused)
  summary.txt             output of sim/summarize.py
  logs/                   system1.log, system2.log, eval.log
```

### Browse results

```bash
sim/.venv/bin/python viewer/server.py --log-dir results/atomic_seen --port 8090   # --host 0.0.0.0 for LAN
# open http://127.0.0.1:8090/   (over SSH: ssh -L 8090:127.0.0.1:8090 <host>)
```

Per-task success rates, and for each episode: the rollout played turn by turn with the System2
subgoal of the running segment, a step timeline against the official horizon (▲ = language-steering
intervention), System2's cold plan, and every turn's System2 input clip, thought, judgement,
subgoal, plan update, language-steering interventions and raw response next to the System1 segment it produced.
It reads the results live (Reload), so it can watch a run in progress.

`sim/summarize.py` averages tasks per split (latest `stats.json` per task, like
robocasa-benchmark/openpi `get_eval_stats.py`). It prints the leaderboard fields
(`atomic_seen_success`, `composite_seen_success`, `composite_unseen_success`) and the
task-weighted overall (18/16/16).

## Evaluation protocol

`sim/official_eval.py` reproduces the environment side of robocasa-benchmark/openpi
`examples/robocasa/main.py` (commit 5a6beda):

| | official client | here |
| --- | --- | --- |
| env | `gym.make(f"robocasa/{task}", split="pretrain", seed=7)`, one env per task | same |
| global seed | `np.random.seed(7)` once per process | same |
| trials | 50 sequential `env.reset()` calls on that env | same (`--num-trials 50`) |
| instruction | `obs["annotation.human.task_description"]` (= `ep_meta["lang"]`) | same field |
| step limit | `get_task_horizon(task)` env steps | same, counted cumulatively over all System1 segments |
| success | `info["success"]` (= `env._check_success()`) after every step; stop at the first success | same check after every step; stop at the first success |
| cameras | left/right agentview + eye-in-hand, 256×256 | same renders |

The one difference is the policy: two servers in a loop instead of one websocket policy. Notes:

- **System2's "Current task status" line** comes from the same success check. The episode ends on
  the first success, so when System2 is queried this line always reads `ongoing`.
- **Early stop.** If System2 declares `task_finish` before the env reports success, the episode
  ends and is scored a failure. The step budget is never extended.
- **Language steering.** `system2/language_steering/sys2_steering*.py` revise System2's output
  (general plus per-task revisions, keyed on the env name and the gripper width) and are on by
  default. Every intervention is recorded in `episode.json`; `-- --no-task-rules --no-general-rules`
  disables them (the flag names are unchanged).
- **Scenes.** With the env seeded, trial 0's scene is fixed. Later trials' scenes also depend on
  how many env steps earlier trials took (stepping consumes the env RNG), exactly as in the
  official client, so keep each task's trials sequential in one env. `main.py` runs all tasks in
  one process with one global seed; `--tasks all` in one process matches that, while parallel
  per-task processes each start from `np.random.seed(7)`.

## Provenance

Copied (not modified unless stated) from these sources:

| here | source |
| --- | --- |
| `system1/src/openpi/**` (except below), `system1/packages/openpi-client` | openpi fork `yinpei-tri/openpi` @ e5bad79a |
| `system1/src/openpi/training/config.py`, `policies/policy_config.py`, `serve_system1.py` | slim rewrites of `training/config.py`, `policies/policy_config.py`, `scripts/serve_policy.py` (+ the tag resolver from `scripts/train.py`); same transforms and model config |
| `sim/combined_eval.py`, `sim/sys2_client.py`, `system2/language_steering/sys2_steering*.py` (were `sys2_rules*.py`), `sim/stop_criterion.py` | openpi fork `examples/robocasa/` @ e5bad79a (only import lines and module names changed) |
| `sim/s1_io.py` | the 8 symbols `combined_eval.py` uses from `examples/robocasa/subtask_eval.py` |
| `sim/subtask_env.py` | robocasa fork `yinpei-tri/robocasa` @ 26a7ba99, `robocasa/scripts/eval/subtask_env.py` |
| `sim/vendor/sys2/data/video_policy.py`, `scripts/serve_system2.sh` flags | `yinpei-tri/sys2_train_eval` @ 189c24b3 |
| `sim/official_eval.py` env lifecycle | robocasa-benchmark/openpi `examples/robocasa/main.py` @ 5a6beda |

System1 parity check: the slim server and the original openpi server, given the same checkpoint
and input, returned bit-identical actions/progress/prompt for a progress-as-action checkpoint.
For a progress-head checkpoint the progress output was identical and actions agreed to within
openpi's own run-to-run GPU nondeterminism (~2e-3).
