from __future__ import annotations

import math

import mujoco
import numpy as np

from humanoid_mujoco.config.g1_config import G1Config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rotation_matrix_from_quat(quat: np.ndarray) -> np.ndarray:
    """MuJoCo quaternion (w, x, y, z) → 3×3 rotation matrix."""
    res = np.zeros(9)
    mujoco.mju_quat2Mat(res, quat)
    return res.reshape(3, 3)


def _base_rot(data: mujoco.MjData) -> np.ndarray:
    """Base body rotation matrix (world → body frame)."""
    return _rotation_matrix_from_quat(data.qpos[3:7])


def projected_gravity(data: mujoco.MjData) -> np.ndarray:
    """Project gravity vector into body frame."""
    R = _base_rot(data)
    gravity_world = np.array([0.0, 0.0, -1.0])
    return R.T @ gravity_world


def base_lin_vel_body(data: mujoco.MjData) -> np.ndarray:
    """Linear velocity in body frame."""
    R = _base_rot(data)
    return R.T @ data.qvel[0:3]


def base_ang_vel_body(data: mujoco.MjData) -> np.ndarray:
    """Angular velocity in body frame."""
    R = _base_rot(data)
    return R.T @ data.qvel[3:6]


# ---------------------------------------------------------------------------
# Reward terms
# ---------------------------------------------------------------------------

def reward_lin_vel_tracking(
    data: mujoco.MjData,
    cmd: np.ndarray,
    config: G1Config,
) -> float:
    """Linear velocity tracking reward (vx, vy)."""
    vel_xy = base_lin_vel_body(data)[:2]
    error = np.sum((cmd[:2] - vel_xy) ** 2)
    return float(math.exp(-error / config.vel_tracking_sigma))


def reward_ang_vel_tracking(
    data: mujoco.MjData,
    cmd: np.ndarray,
    config: G1Config,
) -> float:
    """Yaw rate tracking reward."""
    ang_vel_z = base_ang_vel_body(data)[2]
    error = (cmd[2] - ang_vel_z) ** 2
    return float(math.exp(-error / config.vel_tracking_sigma))


def penalty_lin_vel_z(data: mujoco.MjData) -> float:
    """Penalty for base vertical velocity."""
    return float(data.qvel[2] ** 2)


def penalty_ang_vel_xy(data: mujoco.MjData) -> float:
    """Penalty for base roll/pitch angular velocity."""
    ang_vel = base_ang_vel_body(data)
    return float(ang_vel[0] ** 2 + ang_vel[1] ** 2)


def penalty_orientation(data: mujoco.MjData) -> float:
    """Upright posture penalty (projected gravity xy components)."""
    pg = projected_gravity(data)
    return float(pg[0] ** 2 + pg[1] ** 2)


def penalty_base_height(data: mujoco.MjData, config: G1Config) -> float:
    """Penalty for base height deviation from target."""
    return float((data.qpos[2] - config.target_base_height) ** 2)


def penalty_torques(data: mujoco.MjData) -> float:
    """Actuator force penalty (uses actuator_force under position control)."""
    return float(np.sum(data.actuator_force ** 2))


def penalty_joint_vel(data: mujoco.MjData) -> float:
    """Joint velocity penalty."""
    return float(np.sum(data.qvel[6:] ** 2))


def penalty_action_rate(action: np.ndarray, last_action: np.ndarray) -> float:
    """Penalty for large action changes between consecutive steps."""
    return float(np.sum((action - last_action) ** 2))


def penalty_waist_deviation(model: mujoco.MjModel, data: mujoco.MjData) -> float:
    """Penalty for waist joints deviating from upright (0 rad).

    Prevents the policy from bending the torso backward to absorb momentum
    instead of learning proper core stability.
    """
    total = 0.0
    for name in ("waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid >= 0:
            total += data.qpos[model.jnt_qposadr[jid]] ** 2
    return total


def penalty_ankle_deviation(model: mujoco.MjModel, data: mujoco.MjData) -> float:
    """Penalty for extreme ankle pitch (tiptoe or heel-up).

    Without this, the policy exploits tiptoe stance: the ankle site rises
    above the clearance threshold earning clearance reward, while contact
    detection still fires — rewarding a degenerate non-walking gait.
    """
    total = 0.0
    for name in ("left_ankle_pitch_joint", "right_ankle_pitch_joint"):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid >= 0:
            total += data.qpos[model.jnt_qposadr[jid]] ** 2
    return total


def penalty_soft_dof_pos_limit(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    config: G1Config,
) -> float:
    """Penalty for joints approaching their position limits.

    Joints are normalised to [-1, 1] within [lo, hi]; violations beyond
    soft_dof_pos_limit_factor (default 0.9) incur a quadratic penalty.
    """
    total = 0.0
    factor = config.soft_dof_pos_limit_factor
    for j in range(model.njnt):
        if not model.jnt_limited[j]:
            continue
        lo, hi = model.jnt_range[j]
        half = 0.5 * (hi - lo)
        if half <= 0.0:
            continue
        q = data.qpos[model.jnt_qposadr[j]]
        normalized = (q - 0.5 * (lo + hi)) / half  # ∈ [-1, 1] within limits
        violation = max(0.0, abs(normalized) - factor)
        total += violation ** 2
    return total


def reward_feet_contact_timing(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    phase: float,
) -> float:
    """Foot contact timing reward synced to gait clock.

    phase ∈ [0, π)  → left foot grounded, right foot swinging
    phase ∈ [π, 2π) → right foot grounded, left foot swinging

    Uses a continuous force-proportional reward instead of binary matching
    so the policy receives a gradient signal regardless of contact state.
    """
    contact_threshold = 10.0  # Newtons (normalisation reference)

    left_geoms = _geoms_of_body(model, "left_ankle_roll_link")
    right_geoms = _geoms_of_body(model, "right_ankle_roll_link")

    left_force = _body_contact_force(model, data, left_geoms)
    right_force = _body_contact_force(model, data, right_geoms)

    # Normalised contact intensity ∈ [0, 1]
    left_norm = min(left_force / contact_threshold, 1.0)
    right_norm = min(right_force / contact_threshold, 1.0)

    left_should_contact = phase < math.pi
    right_should_contact = phase >= math.pi

    # Stance foot: reward proportional to contact force
    # Swing foot: reward proportional to *absence* of contact force
    reward = 0.0
    reward += left_norm if left_should_contact else (1.0 - left_norm)
    reward += right_norm if right_should_contact else (1.0 - right_norm)
    return reward * 0.5  # normalise to [0, 1]


def reward_feet_clearance(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    phase: float,
    min_clearance: float = 0.05,
    max_clearance: float = 0.15,
) -> float:
    """Clearance reward: foot site lifts enough during swing phase.

    Clamped to [min_clearance, max_clearance] to prevent the policy from
    exploiting unbounded reward by lifting the foot excessively.
    Uses named sites 'left_foot' / 'right_foot' for height.
    """
    left_site = _find_site_id(model, "left_foot")
    right_site = _find_site_id(model, "right_foot")

    reward = 0.0
    left_in_swing = phase >= math.pi
    right_in_swing = phase < math.pi

    if left_site is not None and left_in_swing:
        height = data.site_xpos[left_site][2]
        reward += min(max(0.0, height - min_clearance), max_clearance)

    if right_site is not None and right_in_swing:
        height = data.site_xpos[right_site][2]
        reward += min(max(0.0, height - min_clearance), max_clearance)

    return reward


# ---------------------------------------------------------------------------
# Total reward
# ---------------------------------------------------------------------------

def compute_reward_terms(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    config: G1Config,
    cmd: np.ndarray,
    action: np.ndarray,
    last_action: np.ndarray,
    phase: float,
) -> dict[str, float]:
    """Return each weighted reward term individually for logging."""
    return {
        "lin_vel":          config.w_lin_vel           *  reward_lin_vel_tracking(data, cmd, config),
        "ang_vel":          config.w_ang_vel           *  reward_ang_vel_tracking(data, cmd, config),
        "alive":            config.w_alive,
        "feet_contact":     config.w_feet_contact      *  reward_feet_contact_timing(model, data, phase),
        "feet_clear":       config.w_feet_clearance    *  reward_feet_clearance(model, data, phase),
        "orientation":      config.w_orientation       * -penalty_orientation(data),
        "base_height":      config.w_base_height       * -penalty_base_height(data, config),
        "lin_vel_z":        config.w_lin_vel_z         * -penalty_lin_vel_z(data),
        "ang_vel_xy":       config.w_ang_vel_xy        * -penalty_ang_vel_xy(data),
        "torques":          config.w_torques           * -penalty_torques(data),
        "joint_vel":        config.w_joint_vel         * -penalty_joint_vel(data),
        "action_rate":      config.w_action_rate       * -penalty_action_rate(action, last_action),
        "dof_limit":        config.w_soft_dof_limit    * -penalty_soft_dof_pos_limit(model, data, config),
        "waist_deviation":  config.w_waist_deviation   * -penalty_waist_deviation(model, data),
        "ankle_deviation":  config.w_ankle_deviation   * -penalty_ankle_deviation(model, data),
    }


def compute_reward(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    config: G1Config,
    cmd: np.ndarray,
    action: np.ndarray,
    last_action: np.ndarray,
    phase: float,
) -> float:
    return sum(compute_reward_terms(model, data, config, cmd, action, last_action, phase).values())


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _geoms_of_body(model: mujoco.MjModel, body_name: str) -> list[int]:
    """Return all geom IDs attached to a named body."""
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id < 0:
        return []
    return [i for i in range(model.ngeom) if model.geom_bodyid[i] == body_id]


def _body_contact_force(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    geom_ids: list[int],
) -> float:
    """Total normal contact force (Newtons) across a set of geoms."""
    if not geom_ids:
        return 0.0
    geom_set = set(geom_ids)
    total = 0.0
    for i in range(data.ncon):
        con = data.contact[i]
        if con.geom1 in geom_set or con.geom2 in geom_set:
            force = np.zeros(6)
            mujoco.mj_contactForce(model, data, i, force)
            total += abs(force[0])
    return total


def _find_site_id(model: mujoco.MjModel, name: str) -> int | None:
    """Return site ID by name, or None if not found."""
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
    return site_id if site_id >= 0 else None
