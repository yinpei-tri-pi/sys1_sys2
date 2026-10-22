"""
Camera-enabled env + observation assembly for POLICY (Phase-2) subtask eval.

Phase-1 (oracle) needs no rendering. Phase-2 rolls out a served pi0.5 System1 policy,
which needs the three native cameras + the raw 16-d proprio state, assembled into the
exact infer-dict the openpi ``RobocasaInputs`` transform expects.

Camera / state contract (verified against openpi src/openpi/policies/robocasa_policy.py):
  images  : scene_left  = robot0_agentview_left
            scene_right = robot0_agentview_right
            wrist       = robot0_eye_in_hand         (all HxWx3 uint8)
  state   : RAW 16-d, concatenated in the LeRobot ``observation.state`` order
            base_position(3) + base_rotation(4, quat) + eef_pos_rel(3) +
            eef_rot_rel(4, quat) + gripper_qpos(2).
            The SERVER converts raw->lean; we send raw + the frame-0 base reference
            (base_pos_ref / base_yaw_ref) so the server's relative base pose matches
            training (see the training doc's Phase-4 note).

The 16-d layout is exactly ``data_annotation/lowdim.npz['state']`` (also the LeRobot
parquet ``observation.state``), so at inference we read the same fields live from the sim
observables and concatenate them in the same order.
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np

# raw robosuite observable keys -> the 16-d state slices (LeRobot observation.state order)
STATE_OBS_KEYS = [
    ("robot0_base_pos", 3),
    ("robot0_base_quat", 4),
    ("robot0_base_to_eef_pos", 3),
    ("robot0_base_to_eef_quat", 4),
    ("robot0_gripper_qpos", 2),
]

# --- Action-ordering bridge (CRITICAL) --------------------------------------------
# The policy emits a 12-d action in LEROBOT column order (matching the training
# producer + robocasa_policy.sim_action_from_lean):
#   [0:4]=base_motion, [4]=control_mode, [5:8]=eef_pos, [8:11]=eef_rot, [11]=gripper_close
# But an env built via robosuite.make (LU.get_env_metadata env_kwargs) steps with the
# HDF5/robosuite-native controller order (what LU.get_episode_actions returns for replay):
#   [0:3]=eef_pos, [3:6]=eef_rot, [6]=gripper_close, [7:11]=base_motion, [11]=control_mode
# Feeding the LeRobot-order vector straight to env.step scrambles eef into base/control
# slots (the sim barely moves). Reorder before stepping. (This is the array-level
# equivalent of robocasa.utils.env_utils.convert_action, which maps the same LeRobot
# layout into the gym dict.)
def lerobot_action_to_sim(a):
    """Reorder a policy 12-d action (LeRobot column order) to robosuite-native order."""
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    out = np.empty(12, dtype=np.float64)
    out[0:3] = a[5:8]     # eef_pos
    out[3:6] = a[8:11]    # eef_rot
    out[6] = a[11]        # gripper_close
    out[7:11] = a[0:4]    # base_motion
    out[11] = a[4]        # control_mode
    return out


# native camera name -> policy image key
CAMERA_TO_KEY = {
    "robot0_agentview_left": "scene_left",
    "robot0_agentview_right": "scene_right",
    "robot0_eye_in_hand": "wrist",
}
CAMERA_NAMES = list(CAMERA_TO_KEY.keys())


def make_camera_env(lerobot_dir: str | Path, camera_h: int = 256, camera_w: int = 256):
    """Build the task env WITH offscreen camera rendering for the 3 policy cameras."""
    import robosuite  # noqa
    import robocasa  # noqa
    import robocasa.utils.lerobot_utils as LU

    env_meta = LU.get_env_metadata(Path(lerobot_dir))
    ek = env_meta["env_kwargs"]
    ek["env_name"] = env_meta["env_name"]
    ek["has_renderer"] = False
    ek["renderer"] = "mjviewer"
    ek["has_offscreen_renderer"] = True
    ek["use_camera_obs"] = True
    ek["camera_names"] = CAMERA_NAMES
    ek["camera_heights"] = camera_h
    ek["camera_widths"] = camera_w
    return robosuite.make(**ek)


def raw_state_from_obs(obs: dict) -> np.ndarray:
    """Assemble the raw 16-d state vector from a robosuite obs dict."""
    parts = []
    for key, dim in STATE_OBS_KEYS:
        v = np.asarray(obs[key], dtype=np.float32).reshape(-1)
        assert v.shape[0] == dim, f"{key}: expected {dim}, got {v.shape[0]}"
        parts.append(v)
    return np.concatenate(parts, axis=0)


def images_from_obs(obs: dict) -> dict:
    """Extract the 3 policy camera images (uint8 HxWx3) from a robosuite obs dict.

    robosuite returns camera obs under ``<cam>_image``, bottom-up; flip to top-down.
    """
    out = {}
    for cam, key in CAMERA_TO_KEY.items():
        img = np.asarray(obs[f"{cam}_image"])
        img = img[::-1]  # robosuite renders upside-down
        if np.issubdtype(img.dtype, np.floating):
            img = (255 * img).astype(np.uint8)
        out[key] = np.ascontiguousarray(img)
    return out


def base_reference(obs: dict) -> tuple[np.ndarray, float]:
    """Frame-0 base reference (base_pos (3,), base_yaw scalar) for relative base pose.

    Must be captured ONCE at the episode start and reused for the whole episode so the
    server's relative base pose matches training.
    """
    base_pos = np.asarray(obs["robot0_base_pos"], dtype=np.float32).reshape(-1)[:3]
    q = np.asarray(obs["robot0_base_quat"], dtype=np.float32).reshape(-1)  # (x,y,z,w)
    x, y, z, w = q[0], q[1], q[2], q[3]
    yaw = float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))
    return base_pos, yaw


# --- Gripper-settle warmup ---------------------------------------------------------
# After reset_to(states[s]) the physical gripper width can lag the recorded state (it was
# mid-open when saved), so the very first policy actions fight a not-fully-open gripper.
# Run a few "settle" steps first: ZERO motion (eef delta + base = 0), but keep the control
# mode and the gripper command from the subtask's first recorded action, so the gripper
# reaches its intended open/close before the policy takes over.
def sim_to_lerobot12(a_sim) -> np.ndarray:
    """Inverse of lerobot_action_to_sim: robosuite-native 12-d -> LeRobot column order 12-d.
    LeRobot: base[0:4], control[4], eef_pos[5:8], eef_rot[8:11], grip[11]."""
    a = np.asarray(a_sim, dtype=np.float64).reshape(-1)
    out = np.empty(12, dtype=np.float64)
    out[0:4] = a[7:11]   # base_motion
    out[4] = a[11]       # control_mode
    out[5:8] = a[0:3]    # eef_pos
    out[8:11] = a[3:6]   # eef_rot
    out[11] = a[6]       # gripper_close
    return out


def sim_action_to_lean11(a_sim) -> np.ndarray:
    """Robosuite-native 12-d action -> lean 11-d (for normalizing against action norm_stats).

    Inverse of lerobot_action_to_sim then drop torso: lean order =
    base_motion[0:3], control[3], eef_pos[4:7], eef_rot[7:10], grip[10]. Native order is
    eef_pos[0:3], eef_rot[3:6], grip[6], base[7:11], control[11]; torso = base_motion[3] = a[10]
    is dropped (identically 0 in teleop)."""
    a = np.asarray(a_sim, dtype=np.float32).reshape(-1)
    return np.array([a[7], a[8], a[9], a[11], a[0], a[1], a[2], a[3], a[4], a[5], a[6]], dtype=np.float32)


def settle_action_from_first(first_sim_action) -> np.ndarray:
    """Build a zero-motion settle action (robosuite-native order) from the subtask's first
    recorded action: eef_pos=0, eef_rot=0, base=0; keep gripper_close[6] + control_mode[11]."""
    a = np.asarray(first_sim_action, dtype=np.float64).reshape(-1)
    out = np.zeros(12, dtype=np.float64)
    out[6] = a[6]    # gripper_close (native order)
    out[11] = a[11]  # control_mode
    return out


# --- Lean state + quantile norm (for the video overlay; mirrors openpi robocasa_policy) --
def _quat_xyzw_to_rot6d(q):
    q = np.asarray(q, dtype=np.float32)
    q = q / (np.linalg.norm(q) + 1e-8)
    x, y, z, w = q
    c0 = np.array([1 - 2 * (y * y + z * z), 2 * (x * y + z * w), 2 * (x * z - y * w)], dtype=np.float32)
    c1 = np.array([2 * (x * y - z * w), 1 - 2 * (x * x + z * z), 2 * (y * z + x * w)], dtype=np.float32)
    return np.concatenate([c0, c1])


def lean_state_from_raw16(raw16, base_pos_ref, base_yaw_ref) -> np.ndarray:
    """Raw 16-d -> lean 14-d (eef_pos3 + eef_rot6d + grip1 + base_xy_rel2 + yaw_sincos2).
    Matches robocasa_policy.lean_state_from_raw(include_base_pose=True)."""
    s = np.asarray(raw16, dtype=np.float32)
    eef_pos = s[7:10]
    eef_6d = _quat_xyzw_to_rot6d(s[10:14])
    grip = np.array([s[14] - s[15]], dtype=np.float32)
    base_xy = s[0:2] - np.asarray(base_pos_ref, dtype=np.float32)[0:2]
    q = s[3:7]
    yaw = np.arctan2(2.0 * (q[3] * q[2] + q[0] * q[1]), 1.0 - 2.0 * (q[1] ** 2 + q[2] ** 2)) - np.float32(base_yaw_ref)
    yaw_sincos = np.array([np.sin(yaw), np.cos(yaw)], dtype=np.float32)
    return np.concatenate([eef_pos, eef_6d, grip, base_xy, yaw_sincos]).astype(np.float32)


def quantile_norm(x, q01, q99):
    """(x - q01)/(q99 - q01 + 1e-6)*2 - 1 over the first len(x) dims. Matches transforms.Normalize."""
    x = np.asarray(x, dtype=np.float32)
    q01 = np.asarray(q01, dtype=np.float32)[: x.shape[-1]]
    q99 = np.asarray(q99, dtype=np.float32)[: x.shape[-1]]
    return (x - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0


def build_prompt_text(task_goal, subgoal, quality, est_length, executed_step, gripper_flag) -> str:
    """The TEXT portion of the pi0.5 prompt, mirroring openpi robocasa_policy.build_prompt +
    the tokenizer scaffold (state ints are shown separately as raw/norm numbers in the GUI).

    Task: <task>; Current Subgoal: <subgoal>
    Quality: <q>; Estimated Length: <n>; Executed Step: <k>
    (Initial State: <14 ints>; Current State: <14 ints>; Current Gripper: <flag>;) Action:
    """
    tg = str(task_goal).strip().lower().replace("\n", " ").rstrip(".!; ")
    sg = str(subgoal).strip().lower().replace("\n", " ")
    line1 = f"Task: {tg}; Current Subgoal: {sg}"
    line2 = f"Quality: {quality}; Estimated Length: {int(est_length)}; Executed Step: {int(executed_step)}"
    line3 = f"Initial State: <14 anchor ints>; Current State: <14 ints>; Current Gripper: {gripper_flag};"
    return line1 + "\n" + line2 + "\n" + line3 + "\nAction:"
