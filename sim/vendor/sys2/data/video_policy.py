"""Explicit System2 video sampling policies shared by training and vLLM evaluation.

qwen-vl-utils 0.0.14 hard-codes its package defaults. ms-swift patches those globals while
initializing a Qwen template, but direct qwen-vl-utils and vLLM calls do not get that behavior.
Keep policy values in our rows and evaluation requests so correctness does not depend on which
initialization path happened to run first.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class VideoPolicy:
    name: str
    fps: float
    min_frames: int
    max_frames: int
    frame_factor: int = 2

    def swift_kwargs(self) -> dict[str, int | float]:
        """Arguments consumed by qwen_vl_utils.fetch_video via SWIFT chat_template_kwargs."""
        return {
            "fps": self.fps,
            "min_frames": self.min_frames,
            "max_frames": self.max_frames,
        }


SYSTEM2_4FPS_32 = VideoPolicy(
    name="system2_4fps_32",
    fps=4.0,
    min_frames=2,
    max_frames=32,
)

# Compatibility policy for a checkpoint intentionally trained with raw qwen-vl-utils 0.0.14
# defaults. The completed System2 checkpoint did NOT use this policy: ms-swift patched its
# training template to the intended 4-FPS/32-frame settings.
LEGACY_QVU_0_0_14 = VideoPolicy(
    name="legacy_qvu_0_0_14",
    fps=2.0,
    min_frames=4,
    max_frames=768,
)

POLICIES = {policy.name: policy for policy in (SYSTEM2_4FPS_32, LEGACY_QVU_0_0_14)}
DEFAULT_VIDEO_POLICY = SYSTEM2_4FPS_32.name


def get_video_policy(name: str) -> VideoPolicy:
    try:
        return POLICIES[name]
    except KeyError as exc:
        raise ValueError(
            f"unknown System2 video policy {name!r}; choose one of {sorted(POLICIES)}"
        ) from exc


def smart_nframes(total_frames: int, source_fps: float, policy: str | VideoPolicy) -> int:
    """Reproduce qwen-vl-utils 0.0.14 ``smart_nframes`` for an explicit policy."""
    policy = get_video_policy(policy) if isinstance(policy, str) else policy
    if total_frames < policy.frame_factor:
        raise ValueError(
            f"video has {total_frames} frames, fewer than frame_factor={policy.frame_factor}"
        )
    if source_fps <= 0:
        raise ValueError(f"source_fps must be positive, got {source_fps}")

    ceil_factor = lambda value: math.ceil(value / policy.frame_factor) * policy.frame_factor
    floor_factor = lambda value: math.floor(value / policy.frame_factor) * policy.frame_factor
    min_frames = ceil_factor(policy.min_frames)
    max_frames = floor_factor(min(policy.max_frames, total_frames))
    nframes = total_frames / source_fps * policy.fps
    nframes = min(min(max(nframes, min_frames), max_frames), total_frames)
    nframes = floor_factor(nframes)
    if not policy.frame_factor <= nframes <= total_frames:
        raise ValueError(
            f"policy {policy.name} produced invalid nframes={nframes} for "
            f"total_frames={total_frames}, source_fps={source_fps}"
        )
    return nframes


def row_video_nframes(row: dict, policy_name: str) -> int | None:
    """Return the per-request vLLM frame count, or ``None`` for a non-video row."""
    videos = row.get("videos") or []
    if not videos:
        return None
    if len(videos) != 1:
        raise ValueError(
            f"per-request frame policy currently supports one video, got {len(videos)}"
        )
    meta = row.get("meta") or {}
    total_frames = meta.get("clip_frames")
    source_fps = meta.get("clip_fps")
    if not isinstance(total_frames, int) or total_frames < 1:
        raise ValueError(
            f"video sample {row.get('_sample_key', '<unknown>')} has no valid meta.clip_frames"
        )
    if not isinstance(source_fps, (int, float)) or source_fps <= 0:
        raise ValueError(
            f"video sample {row.get('_sample_key', '<unknown>')} has no valid meta.clip_fps"
        )
    return smart_nframes(total_frames, float(source_fps), policy_name)
