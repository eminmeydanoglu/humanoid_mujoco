from __future__ import annotations

import math
from dataclasses import dataclass

import mujoco
import numpy as np

from humanoid_mujoco.teleop.commanding import TeleopCommand


@dataclass(slots=True)
class KeyframeInfo:
    stand_id: int | None


def find_keyframe_id(model: mujoco.MjModel, name: str) -> int | None:
    for key_id in range(model.nkey):
        if model.key(key_id).name == name:
            return key_id
    return None


class G1TeleopController:
    """Simple command-to-actuator mapper for Unitree G1 position actuators."""

    def __init__(self, model: mujoco.MjModel):
        self.model = model
        self.actuator_name_to_idx = {model.actuator(i).name: i for i in range(model.nu)}
        self.keyframes = KeyframeInfo(stand_id=find_keyframe_id(model, "stand"))
        self.base_ctrl = self._make_base_ctrl()
        self._validate_required_actuators()

    def _validate_required_actuators(self) -> None:
        required = {
            "left_hip_pitch_joint",
            "right_hip_pitch_joint",
            "left_knee_joint",
            "right_knee_joint",
            "left_hip_roll_joint",
            "right_hip_roll_joint",
            "waist_yaw_joint",
            "left_shoulder_pitch_joint",
            "right_shoulder_pitch_joint",
            "left_shoulder_roll_joint",
            "right_shoulder_roll_joint",
        }
        missing = sorted(required.difference(self.actuator_name_to_idx))
        if missing:
            raise ValueError(f"Model missing required G1 actuators: {missing}")

    def _make_base_ctrl(self) -> np.ndarray:
        if self.keyframes.stand_id is not None:
            return np.array(
                self.model.key_ctrl[self.keyframes.stand_id], dtype=np.float64
            )
        return np.zeros(self.model.nu, dtype=np.float64)

    def _add(self, ctrl: np.ndarray, actuator_name: str, delta: float) -> None:
        idx = self.actuator_name_to_idx.get(actuator_name)
        if idx is None:
            return
        ctrl[idx] += delta

    def compute_ctrl(self, command: TeleopCommand, sim_time: float) -> np.ndarray:
        ctrl = self.base_ctrl.copy()

        forward = max(-1.5, min(1.5, command.forward))
        lateral = max(-1.5, min(1.5, command.lateral))
        yaw = max(-1.5, min(1.5, command.yaw))
        arm_lift_cmd = max(-1.0, min(1.0, command.arm_lift))
        arm_swing_cmd = max(-1.0, min(1.0, command.arm_swing))

        phase = 2.0 * math.pi * 1.4 * sim_time
        swing = math.sin(phase)

        self._add(ctrl, "left_hip_pitch_joint", 0.10 * forward * swing)
        self._add(ctrl, "right_hip_pitch_joint", -0.10 * forward * swing)
        self._add(ctrl, "left_knee_joint", 0.15 * abs(forward) * max(0.0, -swing))
        self._add(ctrl, "right_knee_joint", 0.15 * abs(forward) * max(0.0, swing))

        self._add(ctrl, "left_hip_roll_joint", 0.08 * lateral)
        self._add(ctrl, "right_hip_roll_joint", 0.08 * lateral)
        self._add(ctrl, "waist_yaw_joint", 0.30 * yaw)

        arm_lift = 0.35 * arm_lift_cmd
        arm_swing = 0.25 * arm_swing_cmd * swing
        self._add(ctrl, "left_shoulder_pitch_joint", arm_lift - arm_swing)
        self._add(ctrl, "right_shoulder_pitch_joint", arm_lift + arm_swing)
        self._add(ctrl, "left_shoulder_roll_joint", 0.20 * arm_swing_cmd)
        self._add(ctrl, "right_shoulder_roll_joint", -0.20 * arm_swing_cmd)

        limited = self.model.actuator_ctrllimited.astype(bool)
        low = self.model.actuator_ctrlrange[:, 0]
        high = self.model.actuator_ctrlrange[:, 1]
        ctrl[limited] = np.clip(ctrl[limited], low[limited], high[limited])
        return ctrl
