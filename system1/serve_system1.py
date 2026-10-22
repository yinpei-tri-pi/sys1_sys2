"""Serve one System1 (pi0.5 RoboCasa) checkpoint over the openpi websocket protocol.

Inference-only version of openpi ``scripts/serve_policy.py``: the checkpoint's own ``config.json``
(or its ``<exp>__<tag>`` dir name) selects the exact model/prompt configuration it was trained with.

    CUDA_VISIBLE_DEVICES=0 uv run serve_system1.py --ckpt <run>/269999 --port 8060
"""

import dataclasses
import logging
import socket

import tyro

from openpi.policies import policy_config as _policy_config
from openpi.serving import websocket_policy_server
from openpi.training import config as _config


@dataclasses.dataclass
class Args:
    # Checkpoint step dir (contains params/ and assets/robocasa_system1/norm_stats.json).
    ckpt: str
    port: int = 8060
    # Explicit config name; default resolves it from the checkpoint's config.json.
    config: str = "auto"


def main(args: Args) -> None:
    if args.config in ("", "auto"):
        train_config = _config.resolve_robocasa_config(args.ckpt)
        if train_config is None:
            raise ValueError(f"no RoboCasa config could be resolved from {args.ckpt} (no config.json and "
                             "no settings tag in the dir name); pass --config explicitly")
    else:
        train_config = _config.get_config(args.config)
    logging.info("System1 config: progress_mode=%s use_progress_head=%s progress_as_action=%s",
                 getattr(train_config.model, "progress_mode", None),
                 getattr(train_config.model, "use_progress_head", None),
                 train_config.data.progress_as_action)
    policy = _policy_config.create_trained_policy(train_config, args.ckpt)
    logging.info("Serving %s on port %d (host %s)", args.ckpt, args.port, socket.gethostname())
    websocket_policy_server.WebsocketPolicyServer(
        policy=policy, host="0.0.0.0", port=args.port, metadata=policy.metadata,
    ).serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
