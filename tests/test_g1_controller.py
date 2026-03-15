from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import pytest

from humanoid_mujoco.teleop.commanding import TeleopCommand
from humanoid_mujoco.teleop.g1_controller import G1TeleopController


def _g1_model_path() -> Path:
    return (
        Path(__file__).resolve().parents[1]
        / "mujoco_menagerie"
        / "unitree_g1"
        / "g1.xml"
    )


def _load_g1_model() -> mujoco.MjModel:
    path = _g1_model_path()
    if not path.exists():
        pytest.skip(f"Missing G1 model file: {path}")
    return mujoco.MjModel.from_xml_path(path.as_posix())


def test_controller_uses_valid_base_ctrl_shape() -> None:
    model = _load_g1_model()
    controller = G1TeleopController(model)
    assert controller.base_ctrl.shape == (model.nu,)


def test_controller_output_changes_with_command() -> None:
    model = _load_g1_model()
    controller = G1TeleopController(model)
    base = controller.compute_ctrl(TeleopCommand(), sim_time=0.0)

    cmd = TeleopCommand(
        forward=1.0,
        lateral=0.5,
        yaw=0.5,
        arm_lift=0.5,
        arm_swing=0.5,
    )
    out = controller.compute_ctrl(cmd, sim_time=0.18)

    assert out.shape == (model.nu,)
    assert np.any(np.abs(out - base) > 1e-9)


def test_controller_respects_ctrl_limits() -> None:
    model = _load_g1_model()
    controller = G1TeleopController(model)
    cmd = TeleopCommand(
        forward=10.0,
        lateral=10.0,
        yaw=10.0,
        arm_lift=10.0,
        arm_swing=10.0,
    )
    out = controller.compute_ctrl(cmd, sim_time=1.23)

    limited = model.actuator_ctrllimited.astype(bool)
    low = model.actuator_ctrlrange[:, 0]
    high = model.actuator_ctrlrange[:, 1]

    assert np.all(out[limited] >= low[limited] - 1e-9)
    assert np.all(out[limited] <= high[limited] + 1e-9)


def test_controller_clamps_input_command_values() -> None:
    model = _load_g1_model()
    controller = G1TeleopController(model)

    huge = TeleopCommand(
        forward=99.0,
        lateral=-99.0,
        yaw=99.0,
        arm_lift=99.0,
        arm_swing=-99.0,
    )
    clipped = TeleopCommand(
        forward=1.5,
        lateral=-1.5,
        yaw=1.5,
        arm_lift=1.0,
        arm_swing=-1.0,
    )

    out_huge = controller.compute_ctrl(huge, sim_time=0.37)
    out_clipped = controller.compute_ctrl(clipped, sim_time=0.37)
    assert np.allclose(out_huge, out_clipped)
