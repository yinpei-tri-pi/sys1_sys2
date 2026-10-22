"""Inference-only slice of openpi's ``training/config.py`` for the RoboCasa System1 policy.

The full training config module imports every data loader, optimizer and dataset in openpi. Serving
a checkpoint needs only three things from it, copied here verbatim from openpi (commit e5bad79a):

  * the ``pi05_robocasa_system1`` model config (``Pi0Config`` arguments),
  * ``RoboCasaDataConfig.create`` -> the RobocasaInputs/Outputs + tokenizer transforms,
  * the checkpoint-tag resolver (``scripts/train.py::robocasa_config_from_tag``), which rebuilds the
    exact config a checkpoint was trained with from ``<run>/config.json``.

Training-only fields (shards, shuffle buffer, optimizer, weight loader, ...) are dropped; they never
affect the policy's input/output transforms or the model graph.
"""

import dataclasses
import json
import pathlib
from typing import Any, Literal

import openpi.models.model as _model
import openpi.models.pi0_config as pi0_config
import openpi.models.tokenizer as _tokenizer
import openpi.policies.robocasa_policy as robocasa_policy
import openpi.transforms as _transforms

ModelType = _model.ModelType


@dataclasses.dataclass(frozen=True)
class DataConfig:
    repo_id: str | None = None
    # Directory within the checkpoint's assets/ directory containing the norm stats.
    asset_id: str | None = None
    norm_stats: dict[str, _transforms.NormStats] | None = None
    repack_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    data_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    model_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    use_quantile_norm: bool = False


@dataclasses.dataclass(frozen=True)
class ModelTransformFactory:
    """Creates model transforms for pi0.5 (the PI05 branch of openpi's factory, verbatim)."""

    default_prompt: str | None = None
    # When the anchor/initial state is appended onto the current state, split the discretized
    # state into two labeled prompt segments at this index.
    state_split: int | None = None
    state_split_label: str = "Initial State"
    task_state_sep: str = ", "
    preserve_newlines: bool = False
    use_gripper_flag: bool = False
    include_state: bool = True

    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        if model_config.model_type != _model.ModelType.PI05:
            raise ValueError(f"System1 serving supports pi0.5 only, got {model_config.model_type}")
        assert isinstance(model_config, pi0_config.Pi0Config)
        return _transforms.Group(
            inputs=[
                _transforms.InjectDefaultPrompt(self.default_prompt),
                _transforms.ResizeImages(224, 224),
                _transforms.TokenizePrompt(
                    _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                    discrete_state_input=model_config.discrete_state_input,
                    state_split=self.state_split,
                    state_split_label=self.state_split_label,
                    task_state_sep=self.task_state_sep,
                    preserve_newlines=self.preserve_newlines,
                    use_gripper_flag=self.use_gripper_flag,
                    include_state=self.include_state,
                ),
                # pi0.5 discretizes the state into the prompt text, so it does not
                # need (and should not get) a zero-padded continuous state vector.
                _transforms.PadStatesAndActions(model_config.action_dim, pad_state=False),
            ],
        )


@dataclasses.dataclass(frozen=True)
class RoboCasaDataConfig:
    """System1 RoboCasa data config -- the knobs that shape the policy's prompt and I/O."""

    repo_id: str = "robocasa_system1"
    prompt_source: Literal["subgoal", "subgoal_detail", "milestone"] = "subgoal"
    include_base_pose: bool = True
    use_anchor_images: bool = True
    include_task_goal: bool = True
    include_subgoal: bool = True
    include_state: bool = True
    include_anchor_state: bool = True
    include_conditioning: bool = True
    include_executed_step: bool = True
    include_est_length: bool = True
    include_gripper_flag: bool = True
    # Progress-as-action: the flow-matching head predicts a per-step subgoal-progress dim appended
    # to the action chunk (instead of a separate progress head).
    progress_as_action: bool = False

    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        del assets_dirs  # norm stats are read from the checkpoint's own assets/ directory
        # No DeltaActions/AbsoluteActions: RoboCasa actions are already delta commands.
        data_transforms = _transforms.Group(
            inputs=[
                robocasa_policy.RobocasaInputs(
                    action_dim=model_config.action_dim,
                    model_type=model_config.model_type,
                    include_base_pose=self.include_base_pose,
                    use_anchor_images=self.use_anchor_images,
                    include_task_goal=self.include_task_goal,
                    include_subgoal=self.include_subgoal,
                    include_anchor_state=self.include_anchor_state,
                    include_conditioning=self.include_conditioning,
                    include_executed_step=self.include_executed_step,
                    include_est_length=self.include_est_length,
                    include_gripper_flag=self.include_gripper_flag,
                )
            ],
            outputs=[robocasa_policy.RobocasaOutputs()],
        )
        state_split = (
            robocasa_policy.lean_state_dim(include_base_pose=self.include_base_pose)
            if self.include_anchor_state
            else None
        )
        model_transforms = ModelTransformFactory(
            state_split=state_split,
            state_split_label="Initial State",
            task_state_sep="\n",
            preserve_newlines=True,
            use_gripper_flag=self.include_gripper_flag,
            include_state=self.include_state,
        )(model_config)
        if self.progress_as_action:
            # INPUT: Normalize(11) -> append progress(12) -> PadStatesAndActions(12->32).
            # OUTPUT: SplitProgressAction pops the progress dim to `progress` before Unnormalize.
            lean_dim = robocasa_policy.LEAN_ACTION_DIM
            model_transforms = dataclasses.replace(
                model_transforms,
                inputs=[_transforms.AppendProgressAction(), *model_transforms.inputs],
                outputs=[_transforms.SplitProgressAction(real_dim=lean_dim), *model_transforms.outputs],
            )
        return DataConfig(
            repo_id=self.repo_id,
            asset_id=self.repo_id,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            use_quantile_norm=model_config.model_type != ModelType.PI0,
        )


@dataclasses.dataclass(frozen=True)
class TrainConfig:
    name: str
    model: _model.BaseModelConfig
    data: RoboCasaDataConfig
    assets_base_dir: str = "./assets"
    policy_metadata: dict[str, Any] | None = None

    @property
    def assets_dirs(self) -> pathlib.Path:
        return (pathlib.Path(self.assets_base_dir) / self.name).resolve()


_CONFIGS = [
    TrainConfig(
        name="pi05_robocasa_system1",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=20,
            max_token_len=256,
            image_keys=("scene_left", "scene_right", "wrist"),
            geometric_aug_cameras=("scene_left", "scene_right", "anchor_scene_left", "anchor_scene_right"),
            use_anchor_images=True,
            flow_loss_real_dim=11,
            use_progress_head=True,
            progress_mode="classes",
            progress_num_classes=10,
            progress_loss_weight=1.0,
            progress_hidden=512,
            progress_num_layers=2,
        ),
        data=RoboCasaDataConfig(
            repo_id="robocasa_system1",
            prompt_source="subgoal",
            use_anchor_images=True,
        ),
    ),
]
_CONFIGS_DICT = {config.name: config for config in _CONFIGS}


def get_config(config_name: str) -> TrainConfig:
    if config_name not in _CONFIGS_DICT:
        raise ValueError(f"Config '{config_name}' not found (available: {sorted(_CONFIGS_DICT)})")
    return _CONFIGS_DICT[config_name]


# ---- checkpoint tag -> config (verbatim from openpi scripts/train.py) ---------------------------

_ROBOCASA_PROMPT_DEVIATIONS = [
    # (attr, default, token-when-different). All default TRUE (prompt-content on).
    ("include_conditioning", True, "nocond"),
    ("include_executed_step", True, "noexec"),
    ("include_est_length", True, "noestl"),
    ("include_state", True, "nostate"),
    ("include_anchor_state", True, "noanchorstate"),
    ("include_task_goal", True, "notask"),
    ("include_subgoal", True, "nosubgoal"),
    ("use_anchor_images", True, "noanchor"),  # read from model (data mirrors it)
    ("include_gripper_flag", True, "nogrip"),
]

_PROMPT_SOURCE_TO_GRAN_VERB = {
    "subgoal": ("fine", "simp"),
    "subgoal_detail": ("fine", "rich"),
    "milestone": ("crse", "simp"),
}
_GRAN_VERB_TO_PROMPT_SOURCE = {v: k for k, v in _PROMPT_SOURCE_TO_GRAN_VERB.items()}


def robocasa_config_from_tag(tag: str, base_config_name: str = "pi05_robocasa_system1") -> TrainConfig:
    """Rebuild the exact config from a settings tag (``prog<x>_gran<y>_verb<z>[_<dev>...]``)."""
    if "__" in tag:
        tag = tag.split("__", 1)[1]  # tolerate a full "<exp>__<tag>" dir name
    tokens = tag.split("_")
    cfg = get_config(base_config_name)
    model_over: dict = {}
    data_over: dict = {}

    for tok in tokens:
        if tok.startswith("prog"):
            p = tok[len("prog"):]
            if p == "cls":
                model_over.update(progress_mode="classes", use_progress_head=True)
                data_over.update(progress_as_action=False)
            elif p == "reg":
                model_over.update(progress_mode="continuous", use_progress_head=True)
                data_over.update(progress_as_action=False)
            elif p == "act":
                model_over.update(use_progress_head=False)
                data_over.update(progress_as_action=True)
            elif p == "none":
                model_over.update(use_progress_head=False)
                data_over.update(progress_as_action=False)
        elif tok.startswith("gran") or tok.startswith("verb"):
            pass  # resolved together below
        else:
            for attr, default, token in _ROBOCASA_PROMPT_DEVIATIONS:
                if tok == token:
                    (model_over if attr == "use_anchor_images" else data_over)[attr] = not default
                    break

    gran = next((t[len("gran"):] for t in tokens if t.startswith("gran")), "fine")
    verb = next((t[len("verb"):] for t in tokens if t.startswith("verb")), "simp")
    ps = _GRAN_VERB_TO_PROMPT_SOURCE.get((gran, verb))
    if ps is not None:
        data_over["prompt_source"] = ps

    model = dataclasses.replace(cfg.model, **model_over) if model_over else cfg.model
    data = dataclasses.replace(cfg.data, **data_over) if data_over else cfg.data
    return dataclasses.replace(cfg, model=model, data=data)


def resolve_robocasa_config(checkpoint_dir) -> TrainConfig | None:
    """Config for a checkpoint: ``<run>/config.json`` first, then the ``<exp>__<tag>`` dir name.

    Accepts a step dir (``.../269999``) or the run root; searches both for config.json.
    """
    p = pathlib.Path(str(checkpoint_dir))
    for cand in (p, p.parent):
        cj = cand / "config.json"
        if cj.exists():
            try:
                meta = json.loads(cj.read_text())
            except ValueError:
                continue  # unreadable config.json: fall through to dir-name parsing
            if meta.get("robocasa_tag"):
                base = meta.get("base_config", "pi05_robocasa_system1")
                if base not in _CONFIGS_DICT:
                    # Never fall back silently: a different base config means a different graph.
                    raise ValueError(f"{cj}: base_config {base!r} is not in this inference-only "
                                     f"config (available: {sorted(_CONFIGS_DICT)}); port it from "
                                     "openpi src/openpi/training/config.py")
                return robocasa_config_from_tag(meta["robocasa_tag"], base)
    for part in reversed(p.parts):
        if "__" in part and any(part.split("__", 1)[1].startswith(f"prog{x}") for x in ("cls", "reg", "act", "none")):
            return robocasa_config_from_tag(part)
    return None
