# Results: RoboCasa365 leaderboard protocol, target split

System2 (Qwen3-VL-4B planner) + System1 (pi0.5, progress-as-action) with all language steering on,
evaluated with this repo's `scripts/run_fleet.sh` in RoboCasa365's held-out **target** scenes: the 10
target kitchens and the target object set, as RoboCasa's benchmark documentation specifies. Neither
model saw target-split data in training; both were trained on the pretrain split only.

| Split | Tasks | Successful episodes | Success (%) |
| --- | --- | --- | --- |
| Atomic, seen | 18 | 709 / 900 | 78.8 |
| Composite, seen | 16 | 395 / 800 | 49.4 |
| Composite, unseen | 16 | 181 / 800 | 22.6 |
| **Overall** (task-weighted 18/16/16) | 50 | **1285 / 2500** | **51.4** |

## Setup

| | |
| --- | --- |
| Protocol | robocasa-benchmark/openpi `main.py` environment lifecycle with the target scene split: `gym.make("robocasa/<task>", split="target", seed=7)`, 50 sequential `env.reset()` trials per task, success checked after every step, stop at the first success, per-task registry horizon |
| Episodes | 2500 (50 tasks x 50 trials), 0 infrastructure errors. One trial (ArrangeBreadBasket, trial 10) is scored a failure because System2's plan degenerated into a repeated word until the token limit and no plan could be executed |
| System1 | [`yinpei-tri-pi/robocasa-system1-pi05-progact`](https://huggingface.co/yinpei-tri-pi/robocasa-system1-pi05-progact), step 269999 |
| System2 | [`yinpei-tri-pi/robocasa-system2-qwen3vl-4b`](https://huggingface.co/yinpei-tri-pi/robocasa-system2-qwen3vl-4b), checkpoint-17124, vLLM 0.19.1, greedy decoding, up to 1024 tokens per reply |
| Language steering | mandatory + general + task + last-milestone retry (the default; behavior version 10), on in all 2500 episodes |
| Simulator | official RoboCasa 1.0.1 (`2ce1aec`), robosuite 1.5.2 (`232ce7d4`), MuJoCo 3.3.1, EGL rendering |
| Hardware | 8x A100-SXM4-80GB, 16 stacks (two System1 + System2 + simulator stacks per GPU, `--stacks-per-gpu 2 --s1-mem-frac 0.13 --s2-mem-frac 0.20`), tasks pulled from a shared queue, longest horizon first |
| Compute | 110.3 task-hours. The longest single task (PackIdenticalLunches, 50 sequential trials) takes 6.0 h, which bounds the wall time of any run however many GPUs it uses |
| Wall time | Run in two attempts. The first (tri-hq-ml-dgx-08, 2026-10-28 09:16-13:44) completed 12 tasks before its host process was terminated externally. The second (tri-hq-ml-dgx-12, 2026-10-29 08:08-14:10, 6.0 h) ran the other 38; tasks the stop interrupted were rerun from trial 0, so every task's 50 trials ran in one uninterrupted env |

The 12 first-attempt tasks ran with a 512-token System2 limit; none of their 14378 System2 replies
exceeded 449 tokens, so the limit never applied and the two attempts are equivalent to one run.

Episode endings: 1285 environment success, 1180 hit the horizon, 20 System2 returned no subgoal,
9 stopped by the repeat cap, 5 System2 declared the task finished before the environment did, 1 no plan.

## Per task

The last column is the same model in the training kitchens (`--split pretrain`), for reference.

### Atomic, seen (78.8; pretrain scenes 80.1)

| Task | Success (%) | Horizon | Mean System2 turns | Pretrain scenes (%) |
| --- | --- | --- | --- | --- |
| CloseBlenderLid | 48 | 900 | 10.0 | 42 |
| CloseFridge | 92 | 900 | 4.9 | 96 |
| CloseToasterOvenDoor | 84 | 450 | 3.0 | 92 |
| CoffeeSetupMug | 54 | 600 | 7.2 | 50 |
| NavigateKitchen | 60 | 450 | 5.0 | 74 |
| OpenCabinet | 96 | 1050 | 4.8 | 90 |
| OpenDrawer | 94 | 750 | 4.4 | 80 |
| OpenStandMixerHead | 100 | 450 | 2.3 | 94 |
| PickPlaceCounterToCabinet | 76 | 750 | 8.8 | 74 |
| PickPlaceCounterToStove | 88 | 600 | 7.0 | 86 |
| PickPlaceDrawerToCounter | 74 | 750 | 10.5 | 88 |
| PickPlaceSinkToCounter | 100 | 900 | 6.3 | 92 |
| PickPlaceToasterToCounter | 90 | 600 | 7 | 90 |
| SlideDishwasherRack | 78 | 450 | 3.6 | 86 |
| TurnOffStove | 44 | 750 | 7.0 | 58 |
| TurnOnElectricKettle | 92 | 450 | 3.4 | 92 |
| TurnOnMicrowave | 82 | 450 | 4.5 | 86 |
| TurnOnSinkFaucet | 66 | 600 | 3.1 | 72 |

### Composite, seen (49.4; pretrain scenes 51.5)

| Task | Success (%) | Horizon | Mean System2 turns | Pretrain scenes (%) |
| --- | --- | --- | --- | --- |
| DeliverStraw | 56 | 2550 | 21.0 | 58 |
| GetToastedBread | 74 | 3000 | 13.9 | 56 |
| KettleBoiling | 54 | 1500 | 14.6 | 46 |
| LoadDishwasher | 46 | 1800 | 17.8 | 70 |
| PackIdenticalLunches | 20 | 3900 | 52.7 | 24 |
| PreSoakPan | 70 | 2400 | 15.7 | 62 |
| PrepareCoffee | 12 | 1800 | 15.4 | 18 |
| RinseSinkBasin | 50 | 1350 | 12.3 | 42 |
| ScrubCuttingBoard | 62 | 1200 | 12.3 | 76 |
| SearingMeat | 36 | 4350 | 29.1 | 28 |
| SetUpCuttingStation | 58 | 2400 | 23.3 | 64 |
| StackBowlsCabinet | 76 | 2100 | 11.5 | 72 |
| SteamInMicrowave | 32 | 2100 | 31.2 | 50 |
| StirVegetables | 42 | 2400 | 18.5 | 36 |
| StoreLeftoversInBowl | 40 | 2550 | 22.9 | 60 |
| WashLettuce | 62 | 1650 | 11.0 | 62 |

### Composite, unseen (22.6; pretrain scenes 28.1)

| Task | Success (%) | Horizon | Mean System2 turns | Pretrain scenes (%) |
| --- | --- | --- | --- | --- |
| ArrangeBreadBasket | 42 | 4350 | 44.7 | 32 |
| ArrangeTea | 24 | 2250 | 17.8 | 46 |
| BreadSelection | 50 | 1950 | 15.8 | 62 |
| CategorizeCondiments | 4 | 1650 | 15.3 | 4 |
| CuttingToolSelection | 36 | 1200 | 13.8 | 24 |
| GarnishPancake | 36 | 2700 | 27.9 | 54 |
| GatherTableware | 2 | 2250 | 19.1 | 6 |
| HeatKebabSandwich | 0 | 2700 | 27.1 | 2 |
| MakeIceLemonade | 28 | 3000 | 30.9 | 48 |
| PanTransfer | 2 | 1800 | 27.6 | 0 |
| PortionHotDogs | 10 | 2250 | 36.0 | 4 |
| RecycleBottlesByType | 6 | 2850 | 33.3 | 18 |
| SeparateFreezerRack | 8 | 2400 | 36.0 | 6 |
| WaffleReheat | 56 | 4050 | 35.3 | 48 |
| WashFruitColander | 44 | 3150 | 24.1 | 50 |
| WeighIngredients | 14 | 3000 | 22.0 | 46 |

## Pretrain-scene reference run

An earlier 2500-episode run followed the official client's default `split="pretrain"`, i.e. the
2500 training kitchens with the pretraining objects. That is not the benchmark setting; its scores
(80.1 / 51.5 / 28.1, overall 54.3) appear above only as a reference column. The gap is largest on
the composite-unseen tasks, which lose 5.5 points in the held-out kitchens.

A third check replayed 30 recorded target-split episodes per task (`--protocol target`, 1500
episodes): 54.5% with the same checkpoints, against 53.7% in the training repo.

## Reproducing

```bash
bash scripts/setup.sh
bash scripts/download_checkpoints.sh
bash scripts/run_fleet.sh --stacks-per-gpu 2 --s1-mem-frac 0.13 --s2-mem-frac 0.20   # target split, 8 x 80 GB GPUs
```

Success rates vary by a few points between runs: System1 samples its flow-matching noise on the GPU,
so episodes diverge even with fixed scenes.
