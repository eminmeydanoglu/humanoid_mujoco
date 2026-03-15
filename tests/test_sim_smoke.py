from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import pytest

from humanoid_mujoco.teleop.commanding import TeleopCommand
from humanoid_mujoco.teleop.g1_controller import G1TeleopController


def test_g1_sim_smoke_runs_multiple_steps() -> None:
    scene_path = (
        Path(__file__).resolve().parents[1]
        / "mujoco_menagerie"
        / "unitree_g1"
        / "scene.xml"
    )
    if not scene_path.exists():
        pytest.skip(f"Missing G1 scene file: {scene_path}")
    model = mujoco.MjModel.from_xml_path(scene_path.as_posix())
    data = mujoco.MjData(model)
    controller = G1TeleopController(model)

    cmd = TeleopCommand(forward=0.4, yaw=0.2, arm_swing=0.4)
    for _ in range(1500):
        data.ctrl[:] = controller.compute_ctrl(cmd, data.time)
        mujoco.mj_step(model, data)

    assert data.time > 0.0
    assert np.isfinite(data.qpos).all()
    assert np.isfinite(data.qvel).all()
