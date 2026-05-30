from __future__ import annotations

import math
from typing import Any

import mujoco
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from humanoid_mujoco.config.g1_config import G1Config
from humanoid_mujoco.rewards.reward_functions import compute_reward, compute_reward_terms
from humanoid_mujoco.teleop.g1_controller import find_keyframe_id


class G1LocomotionEnv(gym.Env):
    """Unitree G1 velocity-tracked locomotion environment.

    Observation: [base_lin_vel(3), base_ang_vel(3), projected_gravity(3),
                  velocity_command(3), joint_positions(nj), joint_velocities(nj),
                  last_action(nu), clock_signal(2)]

    Action: normalised position targets ∈ [-1, 1]^nu
            ctrl = stand_ctrl + action_scale * action
    """

    metadata = {"render_modes": []}

    def __init__(self, config: G1Config | None = None) -> None:
        super().__init__()
        self.config = config or G1Config()

        mjcf = str(self.config.mjcf_path)
        self.model = mujoco.MjModel.from_xml_path(mjcf)
        self.data = mujoco.MjData(self.model)
        sim_dt = float(self.model.opt.timestep)
        self._control_substeps = max(1, int(round(self.config.dt / sim_dt)))
        self.control_dt = self._control_substeps * sim_dt
        if not math.isclose(self.control_dt, self.config.dt, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError(
                f"config.dt={self.config.dt} is not an integer multiple of model timestep={sim_dt}"
            )

        # Core dimensions
        self._nj = self.model.nv - 6       # joint DOF (excluding floating base)
        self._nu = self.model.nu            # number of actuators
        obs_dim = 3 + 3 + 3 + 3 + self._nj + self._nj + self._nu + 2

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self._nu,), dtype=np.float32
        )

        # Stand keyframe
        self._stand_id = find_keyframe_id(self.model, "stand")
        self._base_ctrl = self._load_base_ctrl()
        self._default_qpos = self._load_default_qpos()

        # Override PD gains: gainprm[0]=kp (feed-forward), biasprm[1]=-kp (feedback spring).
        # Both must match; changing only one creates a spurious position-dependent force.
        # biasprm[2] is the derivative gain — rescale proportionally to preserve damping ratio.
        kp_old = 500.0
        kp_new = self.config.kp
        self.model.actuator_gainprm[:, 0] = kp_new
        self.model.actuator_biasprm[:, 1] = -kp_new
        self.model.actuator_biasprm[:, 2] *= math.sqrt(kp_new / kp_old)

        # Base body id for domain randomization
        self._base_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        if self._base_body_id < 0:
            self._base_body_id = 1  # fallback: first non-world body

        # Cache original values for domain randomization (prevents accumulation across resets)
        self._orig_geom_friction = self.model.geom_friction[:, 0].copy()
        self._orig_body_mass = self.model.body_mass[self._base_body_id]

        # Episode state
        self._step_count = 0
        self._phase = 0.0
        self._cmd = np.zeros(3, dtype=np.float64)
        self._last_action = np.zeros(self._nu, dtype=np.float32)
        self._rng = np.random.default_rng()

    # ------------------------------------------------------------------
    # gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        # Reset to stand keyframe
        if self._stand_id is not None:
            mujoco.mj_resetDataKeyframe(self.model, self.data, self._stand_id)
        else:
            mujoco.mj_resetData(self.model, self.data)
        self.data.xfrc_applied[:] = 0.0

        # Domain randomization
        self._apply_domain_randomization()

        # Initial state noise
        self.data.qpos[7:] += self._rng.normal(0, 0.02, self._nj)
        self.data.qvel[6:] += self._rng.normal(0, 0.1, self._nj)
        mujoco.mj_forward(self.model, self.data)

        # Sample velocity command
        self._cmd = self._sample_command()

        # Reset episode state
        self._step_count = 0
        self._phase = self._rng.uniform(0, 2 * math.pi)
        self._last_action = np.zeros(self._nu, dtype=np.float32)

        return self._get_obs(), {}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        action = np.asarray(action, dtype=np.float32)

        # Compute and apply position target
        ctrl = self._base_ctrl + self.config.action_scale * action
        lo = self.model.actuator_ctrlrange[:, 0]
        hi = self.model.actuator_ctrlrange[:, 1]
        limited = self.model.actuator_ctrllimited.astype(bool)
        ctrl[limited] = np.clip(ctrl[limited], lo[limited], hi[limited])
        self.data.ctrl[:] = ctrl

        apply_push = (
            self.config.push_enabled
            and self.config.push_interval_steps > 0
            and (self._step_count + 1) % self.config.push_interval_steps == 0
        )
        if apply_push:
            self._apply_push()

        for _ in range(self._control_substeps):
            mujoco.mj_step(self.model, self.data)

        if apply_push:
            self.data.xfrc_applied[self._base_body_id, :3] = 0.0

        # Advance gait clock
        self._phase = (self._phase + 2 * math.pi * self.control_dt / self.config.gait_period) % (2 * math.pi)
        self._step_count += 1

        reward_terms = compute_reward_terms(
            self.model, self.data, self.config,
            self._cmd, action, self._last_action, self._phase,
        )
        reward = sum(reward_terms.values())
        self._last_action = action.copy()

        terminated = self._is_terminated()
        truncated = self._step_count >= self.config.max_episode_steps

        obs = self._get_obs()
        return obs, reward, terminated, truncated, {"reward_terms": reward_terms}

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        from humanoid_mujoco.rewards.reward_functions import (
            base_lin_vel_body,
            base_ang_vel_body,
            projected_gravity,
        )
        lin_vel = base_lin_vel_body(self.data).astype(np.float32)
        ang_vel = base_ang_vel_body(self.data).astype(np.float32)
        pg = projected_gravity(self.data).astype(np.float32)
        cmd = self._cmd.astype(np.float32)

        joint_pos = (self.data.qpos[7:] - self._default_qpos[7:]).astype(np.float32)
        joint_vel = self.data.qvel[6:].astype(np.float32)
        last_act = self._last_action

        clock = np.array([math.sin(self._phase), math.cos(self._phase)], dtype=np.float32)

        return np.concatenate([lin_vel, ang_vel, pg, cmd, joint_pos, joint_vel, last_act, clock])

    # ------------------------------------------------------------------
    # Termination check
    # ------------------------------------------------------------------

    def _is_terminated(self) -> bool:
        base_height = self.data.qpos[2]
        if base_height < self.config.min_base_height:
            return True

        # Derive roll and pitch from projected gravity
        from humanoid_mujoco.rewards.reward_functions import projected_gravity
        pg = projected_gravity(self.data)
        roll = math.atan2(pg[1], -pg[2])
        pitch = math.atan2(-pg[0], math.sqrt(pg[1] ** 2 + pg[2] ** 2))

        return abs(roll) > self.config.max_roll or abs(pitch) > self.config.max_pitch

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _load_base_ctrl(self) -> np.ndarray:
        if self._stand_id is not None:
            return np.array(self.model.key_ctrl[self._stand_id], dtype=np.float64)
        return np.zeros(self.model.nu, dtype=np.float64)

    def _load_default_qpos(self) -> np.ndarray:
        if self._stand_id is not None:
            return np.array(self.model.key_qpos[self._stand_id], dtype=np.float64)
        return np.array(self.data.qpos, dtype=np.float64)

    def _sample_command(self) -> np.ndarray:
        vx = self._rng.uniform(*self.config.cmd_vx_range)
        vy = self._rng.uniform(*self.config.cmd_vy_range)
        yaw = self._rng.uniform(*self.config.cmd_yaw_range)
        return np.array([vx, vy, yaw], dtype=np.float64)

    def _apply_domain_randomization(self) -> None:
        # Re-sample from original values each reset to prevent accumulation
        scale = self._rng.uniform(*self.config.friction_range)
        self.model.geom_friction[:, 0] = self._orig_geom_friction * scale

        mass_delta = self._rng.uniform(*self.config.mass_offset_range)
        self.model.body_mass[self._base_body_id] = self._orig_body_mass + mass_delta

    def _apply_push(self) -> None:
        force = self._rng.uniform(
            -self.config.push_force_range,
            self.config.push_force_range,
            size=3,
        )
        self.data.xfrc_applied[self._base_body_id, :3] = force

    def update_config(self, **kwargs: Any) -> None:
        """Update config fields at runtime (used by curriculum callback)."""
        for key, val in kwargs.items():
            object.__setattr__(self.config, key, val)
