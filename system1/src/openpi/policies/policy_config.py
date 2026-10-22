"""Build a served System1 policy from a JAX checkpoint (inference-only slice of openpi's
``policies/policy_config.py``, JAX path verbatim; the PyTorch path is not used for RoboCasa)."""

import dataclasses
import json
import logging
import os
import pathlib
from typing import Any

import jax.numpy as jnp
import numpy as np

import openpi.models.model as _model
import openpi.policies.policy as _policy
import openpi.shared.download as download
import openpi.shared.normalize as _normalize
from openpi.training import config as _config
import openpi.transforms as transforms


def create_trained_policy(
    train_config: _config.TrainConfig,
    checkpoint_dir: pathlib.Path | str,
    *,
    sample_kwargs: dict[str, Any] | None = None,
    default_prompt: str | None = None,
) -> _policy.Policy:
    checkpoint_dir = download.maybe_download(str(checkpoint_dir))
    if os.path.exists(os.path.join(checkpoint_dir, "model.safetensors")):
        raise ValueError(f"{checkpoint_dir} is a PyTorch checkpoint; System1 serving loads JAX params/")

    logging.info("Loading model...")
    model = train_config.model.load(_model.restore_params(checkpoint_dir / "params", dtype=jnp.bfloat16))
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    norm_stats_dir = pathlib.Path(checkpoint_dir) / "assets" / data_config.asset_id
    norm_stats = _normalize.load(norm_stats_dir)
    logging.info(f"Loaded norm stats from {norm_stats_dir}")

    # `noanchorstate` checkpoints were trained with a 14-d state but may ship a tiled 28-d stats
    # entry; keep only the current-state half (identical to openpi's policy_config).
    rc_tag = ""
    for cand in (pathlib.Path(str(checkpoint_dir)), pathlib.Path(str(checkpoint_dir)).parent):
        cj = cand / "config.json"
        if cj.exists():
            try:
                rc_tag = json.loads(cj.read_text()).get("robocasa_tag", "") or ""
            except ValueError:
                rc_tag = ""
            break
    rc_tag = rc_tag or str(checkpoint_dir)  # fall back to the dir name (carries the tag)
    if "noanchorstate" in rc_tag and norm_stats.get("state") is not None:
        ss = norm_stats["state"]
        cur = ss.mean.shape[-1] if getattr(ss, "mean", None) is not None else None
        if cur is not None and cur % 2 == 0:
            half = cur // 2
            if np.allclose(ss.mean[:half], ss.mean[half:]):  # genuinely a tiled 28-d entry
                _half = lambda a: (a[:half] if a is not None else a)  # noqa: E731
                norm_stats = dict(norm_stats)
                norm_stats["state"] = transforms.NormStats(
                    mean=_half(ss.mean), std=_half(ss.std), q01=_half(ss.q01), q99=_half(ss.q99))

    # Serving echoes the exact prompt text the tokenizer saw back to the client.
    model_input_transforms = [
        dataclasses.replace(t, emit_prompt_text=True) if isinstance(t, transforms.TokenizePrompt) else t
        for t in data_config.model_transforms.inputs
    ]
    return _policy.Policy(
        model,
        transforms=[
            transforms.InjectDefaultPrompt(default_prompt),
            *data_config.data_transforms.inputs,
            transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *model_input_transforms,
        ],
        output_transforms=[
            *data_config.model_transforms.outputs,
            transforms.Unnormalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.data_transforms.outputs,
        ],
        sample_kwargs=sample_kwargs,
        metadata=train_config.policy_metadata,
        is_pytorch=False,
        pytorch_device=None,
    )

