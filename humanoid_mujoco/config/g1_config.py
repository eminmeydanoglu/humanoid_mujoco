from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


def _default_mjcf() -> Path:
    return Path(__file__).resolve().parents[2] / "mujoco_menagerie" / "unitree_g1" / "scene.xml"


@dataclass(slots=True)
class G1Config:
    # File paths
    mjcf_path: Path = field(default_factory=_default_mjcf)

    # Simulation
    dt: float = 0.02            # 50 Hz
    gait_period: float = 0.8    # seconds
    action_scale: float = 0.1   # position target scale (kp=500 → max torque ~50 N⋅m)

    # Episode
    max_episode_steps: int = 1000  # 20 seconds (50 Hz × 1000)

    # Velocity command ranges (start at curriculum stage 0, widen across stages)
    cmd_vx_range: tuple[float, float] = (0.3, 0.8)
    cmd_vy_range: tuple[float, float] = (0.0, 0.0)
    cmd_yaw_range: tuple[float, float] = (0.0, 0.0)

    # Termination thresholds
    min_base_height: float = 0.3
    max_roll: float = 0.785   # 45 degrees (radians)
    max_pitch: float = 0.785

    # Reward weights
    w_lin_vel: float = 2.0
    w_ang_vel: float = 1.0
    w_alive: float = 0.5
    w_orientation: float = 0.2
    w_base_height: float = 0.1
    w_lin_vel_z: float = 0.5
    w_ang_vel_xy: float = 0.05
    w_torques: float = 0.0002
    w_joint_vel: float = 0.0001
    w_action_rate: float = 0.005
    w_feet_contact: float = 1.0
    w_feet_clearance: float = 0.5
    w_soft_dof_limit: float = 1.0
    soft_dof_pos_limit_factor: float = 0.9
    vel_tracking_sigma: float = 0.25

    # Target base height
    target_base_height: float = 0.78  # metres

    # Domain randomization
    friction_range: tuple[float, float] = (0.3, 1.5)
    mass_offset_range: tuple[float, float] = (-2.0, 2.0)
    push_enabled: bool = False
    push_interval_steps: int = 300
    push_force_range: float = 50.0  # Newtons


# Curriculum stages: each dict sets (cmd_vx_range, cmd_vy_range, cmd_yaw_range, push_enabled)
CURRICULUM_STAGES: list[dict] = [
    # Stage 0: forward only (non-zero to prevent standing-still optimum)
    dict(cmd_vx_range=(0.3, 0.8), cmd_vy_range=(0.0, 0.0), cmd_yaw_range=(0.0, 0.0), push_enabled=False),
    # Stage 1: omnidirectional
    dict(cmd_vx_range=(-0.3, 0.8), cmd_vy_range=(-0.2, 0.2), cmd_yaw_range=(-0.3, 0.3), push_enabled=False),
    # Stage 2: full command range
    dict(cmd_vx_range=(-0.3, 0.8), cmd_vy_range=(-0.3, 0.3), cmd_yaw_range=(-0.5, 0.5), push_enabled=False),
    # Stage 3: external push forces enabled
    dict(cmd_vx_range=(-0.3, 0.8), cmd_vy_range=(-0.3, 0.3), cmd_yaw_range=(-0.5, 0.5), push_enabled=True),
]

CURRICULUM_REWARD_THRESHOLD = 5.0  # advance to next stage when mean reward exceeds this
