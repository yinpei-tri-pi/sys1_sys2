"""System1 request/response helpers used by the closed loop.

Extracted VERBATIM from openpi ``examples/robocasa/subtask_eval.py`` (commit e5bad79a) -- only the
symbols ``combined_eval.py`` uses (it imports this module as ``SE``). The rest of subtask_eval.py is
the standalone per-subtask oracle evaluation, which the closed loop never runs.
"""

import numpy as np
from openpi_client import image_tools

from subtask_env import images_from_obs
from subtask_env import raw_state_from_obs

HORIZON = 20            # policy action chunk length (pi05 System1)


# gripper_close index in ROBOSUITE-NATIVE order (after lerobot_action_to_sim): eef_pos[0:3],
# eef_rot[3:6], grip[6], base[7:11], control[11]. gripper_flag reads the last commanded value
# here, matching training (which read the raw command's gripper dim).
SIM_GRIP_IDX = 6


SIM_CTRL_IDX = 11       # control_mode / base_mode dim in robosuite-native order (>0 = mobile/base mode)


def _resize(img, size):
    return image_tools.convert_to_uint8(image_tools.resize_with_pad(img, size, size))


def _obs_dict(env, *, base_pos_ref, base_yaw_ref, anchor_imgs, anchor_state,
              subgoal_text, task_goal, est_length, executed_step, gripper_flag, resize, obs=None):
    """Assemble the FULL infer dict the served RobocasaInputs transform expects.

    Includes anchor images (subgoal-start), anchor raw state, base ref, and the
    decision-transformer conditioning fields — exactly the training contract.
    """
    if obs is None:
        obs = env._get_observations(force_update=True)
    imgs = images_from_obs(obs)
    element = {
        # current views
        "observation/scene_left": _resize(imgs["scene_left"], resize),
        "observation/scene_right": _resize(imgs["scene_right"], resize),
        "observation/wrist": _resize(imgs["wrist"], resize),
        # anchor (subgoal-start) views — REQUIRED for the anchor config
        "observation/anchor_scene_left": _resize(anchor_imgs["scene_left"], resize),
        "observation/anchor_scene_right": _resize(anchor_imgs["scene_right"], resize),
        "observation/anchor_wrist": _resize(anchor_imgs["wrist"], resize),
        # state: RAW 16-d (server -> lean); anchor raw state + frame-0 base ref
        "observation/state": raw_state_from_obs(obs),
        "observation/anchor_state": np.asarray(anchor_state, dtype=np.float32),
        "observation/base_pos_ref": np.asarray(base_pos_ref, dtype=np.float32),
        "observation/base_yaw_ref": np.float32(base_yaw_ref),
        # prompt text + conditioning (all consumed by build_prompt)
        "prompt": subgoal_text,
        "task_goal": task_goal,
        "quality": "Success",
        "est_length": int(est_length),
        "executed_step": int(executed_step),
        "gripper_flag": gripper_flag,
    }
    return element


def _read_progress(result: dict) -> dict:
    """Normalize whatever progress signal the served policy returned into a summary.

    - progress-as-action: ``result["progress"]`` = per-horizon-step [0,1] array.
    - classes head: ``result["progress_head"]`` = softmax over K deciles -> argmax bucket +
      expected fraction.
    - continuous head: ``result["progress_head"]`` = scalar [0,1].
    """
    out = {}
    if "progress" in result:  # progress-as-action: per-horizon-step [0,1] chunk (12th action dim)
        p = np.asarray(result["progress"]).reshape(-1)
        out["progress_kind"] = "action"
        out["progress_now"] = float(p[0])
        out["progress_end"] = float(p[-1])
        out["progress_chunk"] = p.tolist()  # full per-step progress = the model's 12th action dim
    if "progress_head" in result:
        ph = np.asarray(result["progress_head"]).reshape(-1)
        if ph.size > 1:  # classes softmax
            k = ph.size
            out["progress_kind"] = "classes"
            out["progress_num_classes"] = int(k)
            out["progress_argmax"] = int(np.argmax(ph))       # predicted decile class 0..k-1
            out["progress_conf"] = float(np.max(ph))          # softmax confidence of the argmax
            out["progress_expected_frac"] = float(np.dot(np.arange(k), ph) / (k - 1))
        else:            # continuous scalar
            out["progress_kind"] = "continuous"
            out["progress_now"] = float(ph[0])
    return out


def _stacked_from_obs(obs):
    """3 cam views (native res) stacked side-by-side into one RGB frame, from an obs dict."""
    imgs = images_from_obs(obs)  # already top-down uint8
    return np.concatenate([imgs["scene_left"], imgs["scene_right"], imgs["wrist"]], axis=1)


def _progress_str(prog: dict | None) -> str:
    if not prog:
        return "-"
    k = prog.get("progress_kind")
    if k == "classes":
        return f"class {prog['progress_argmax']}/9 (E[frac]={prog['progress_expected_frac']:.2f})"
    if k == "continuous":
        return f"{prog['progress_now']:.3f} (continuous)"
    if k == "action":
        return f"now={prog['progress_now']:.3f} end={prog['progress_end']:.3f} (as-action)"
    return "-"
