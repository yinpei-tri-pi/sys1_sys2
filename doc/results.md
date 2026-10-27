# Reference results: RoboCasa365 leaderboard protocol

System2 (Qwen3-VL-4B planner) + System1 (pi0.5, progress-as-action) with all language steering on,
evaluated with this repo's `scripts/run_fleet.sh`. The run used the fixed task split (`--schedule
shard`) and predates the renaming of the steering code; the episode loop and steering logic are the
same as today's default.

| Split | Tasks | Success (%) |
| --- | --- | --- |
| Atomic, seen | 18 | 80.1 |
| Composite, seen | 16 | 51.5 |
| Composite, unseen | 16 | 28.1 |
| **Overall** (task-weighted 18/16/16) | 50 | **54.3** |

## Setup

| | |
| --- | --- |
| Protocol | robocasa-benchmark/openpi `main.py`: `gym.make("robocasa/<task>", split="pretrain", seed=7)`, 50 sequential `env.reset()` trials per task, success checked after every step, stop at the first success, per-task registry horizon |
| Episodes | 2500 (50 tasks x 50 trials), 0 infrastructure errors |
| System1 | [`yinpei-tri-pi/robocasa-system1-pi05-progact`](https://huggingface.co/yinpei-tri-pi/robocasa-system1-pi05-progact), step 269999 |
| System2 | [`yinpei-tri-pi/robocasa-system2-qwen3vl-4b`](https://huggingface.co/yinpei-tri-pi/robocasa-system2-qwen3vl-4b), checkpoint-17124, vLLM 0.19.1, greedy decoding |
| Language steering | mandatory + general + task + last-milestone retry (the default; behavior version 10) |
| Simulator | official RoboCasa 1.0.1 (`2ce1aec`), robosuite 1.5.2 (`232ce7d4`), MuJoCo 3.3.1, EGL rendering |
| Hardware | 8x A100-SXM4-80GB, one System1 + System2 + simulator stack per GPU |
| Wall time | 16.5 h (2026-10-25 07:51 to 2026-10-26 00:20), 111 task-hours; on GPUs shared with other jobs, so a dedicated 8-GPU node is faster (a 100-episode pilot, `--num-trials 2`, took 35 min) |

Episode endings: 1358 environment success, 1113 hit the horizon, 15 System2 returned no subgoal,
10 stopped by the repeat cap, 4 System2 declared the task finished before the environment did.

## Per task

### Atomic, seen (80.1)

| Task | Success (%) | Horizon | Mean System2 turns |
| --- | --- | --- | --- |
| CloseBlenderLid | 42 | 900 | 9.9 |
| CloseFridge | 96 | 900 | 4.4 |
| CloseToasterOvenDoor | 92 | 450 | 2.5 |
| CoffeeSetupMug | 50 | 600 | 7.5 |
| NavigateKitchen | 74 | 450 | 3.9 |
| OpenCabinet | 90 | 1050 | 6.1 |
| OpenDrawer | 80 | 750 | 5.1 |
| OpenStandMixerHead | 94 | 450 | 2.4 |
| PickPlaceCounterToCabinet | 74 | 750 | 8.8 |
| PickPlaceCounterToStove | 86 | 600 | 6.8 |
| PickPlaceDrawerToCounter | 88 | 750 | 9.8 |
| PickPlaceSinkToCounter | 92 | 900 | 6.9 |
| PickPlaceToasterToCounter | 90 | 600 | 6.8 |
| SlideDishwasherRack | 86 | 450 | 3.2 |
| TurnOffStove | 58 | 750 | 6.2 |
| TurnOnElectricKettle | 92 | 450 | 3.4 |
| TurnOnMicrowave | 86 | 450 | 4.0 |
| TurnOnSinkFaucet | 72 | 600 | 2.9 |

### Composite, seen (51.5)

| Task | Success (%) | Horizon | Mean System2 turns |
| --- | --- | --- | --- |
| DeliverStraw | 58 | 2550 | 20.8 |
| GetToastedBread | 56 | 3000 | 15.7 |
| KettleBoiling | 46 | 1500 | 14.7 |
| LoadDishwasher | 70 | 1800 | 15.2 |
| PackIdenticalLunches | 24 | 3900 | 50.9 |
| PreSoakPan | 62 | 2400 | 16.5 |
| PrepareCoffee | 18 | 1800 | 14.1 |
| RinseSinkBasin | 42 | 1350 | 12.9 |
| ScrubCuttingBoard | 76 | 1200 | 10.3 |
| SearingMeat | 28 | 4350 | 28.6 |
| SetUpCuttingStation | 64 | 2400 | 21.9 |
| StackBowlsCabinet | 72 | 2100 | 11.3 |
| SteamInMicrowave | 50 | 2100 | 29.2 |
| StirVegetables | 36 | 2400 | 22.0 |
| StoreLeftoversInBowl | 60 | 2550 | 21.3 |
| WashLettuce | 62 | 1650 | 11.6 |

### Composite, unseen (28.1)

| Task | Success (%) | Horizon | Mean System2 turns |
| --- | --- | --- | --- |
| ArrangeBreadBasket | 32 | 4350 | 50.7 |
| ArrangeTea | 46 | 2250 | 16.4 |
| BreadSelection | 62 | 1950 | 15.7 |
| CategorizeCondiments | 4 | 1650 | 15.1 |
| CuttingToolSelection | 24 | 1200 | 14.3 |
| GarnishPancake | 54 | 2700 | 28.5 |
| GatherTableware | 6 | 2250 | 18.7 |
| HeatKebabSandwich | 2 | 2700 | 27.8 |
| MakeIceLemonade | 48 | 3000 | 29.9 |
| PanTransfer | 0 | 1800 | 27.5 |
| PortionHotDogs | 4 | 2250 | 38.2 |
| RecycleBottlesByType | 18 | 2850 | 29.6 |
| SeparateFreezerRack | 6 | 2400 | 36.1 |
| WaffleReheat | 48 | 4050 | 31.6 |
| WashFruitColander | 50 | 3150 | 23.6 |
| WeighIngredients | 46 | 3000 | 16.9 |

## Reproducing

```bash
bash scripts/setup.sh
bash scripts/download_checkpoints.sh
bash scripts/run_fleet.sh          # writes results/official__s1-progact270k_s2-qwen3vl-4b-full-ep3-17124/
```

Success rates vary by a few points between runs: System1 samples its flow-matching noise on the GPU,
so episodes diverge even with fixed scenes. A 30-episodes-per-task run on the recorded target-split
scenes (`--protocol target`) gave 54.5% with the same checkpoints, against 53.7% in the training repo.
