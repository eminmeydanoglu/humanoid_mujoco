from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from humanoid_mujoco.config.g1_config import G1Config
from humanoid_mujoco.envs.g1_env import G1LocomotionEnv


def _make_env() -> G1LocomotionEnv:
    scene = (
        Path(__file__).resolve().parents[1]
        / "mujoco_menagerie"
        / "unitree_g1"
        / "scene.xml"
    )
    if not scene.exists():
        pytest.skip(f"G1 scene file not found: {scene}")
    return G1LocomotionEnv(G1Config(mjcf_path=scene))


def test_env_reset_returns_valid_obs() -> None:
    env = _make_env()
    obs, info = env.reset(seed=42)

    assert obs.shape == env.observation_space.shape
    assert obs.dtype == np.float32
    assert np.isfinite(obs).all(), "reset obs contains inf/nan"


def test_observation_space_matches_obs() -> None:
    env = _make_env()
    obs, _ = env.reset(seed=0)

    assert env.observation_space.contains(obs), (
        f"obs outside space bounds: shape={obs.shape}, "
        f"space shape={env.observation_space.shape}"
    )


def test_env_step_zero_action() -> None:
    env = _make_env()
    env.reset(seed=0)

    action = np.zeros(env.action_space.shape, dtype=np.float32)
    obs, reward, terminated, truncated, info = env.step(action)

    assert obs.shape == env.observation_space.shape
    assert np.isfinite(obs).all(), "step obs contains inf/nan"
    assert np.isfinite(reward), f"reward is not finite: {reward}"
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)


def test_env_step_advances_config_dt() -> None:
    env = _make_env()
    env.reset(seed=0)

    start_time = env.data.time
    action = np.zeros(env.action_space.shape, dtype=np.float32)
    env.step(action)

    assert env.data.time - start_time == pytest.approx(env.config.dt)


def test_env_step_random_actions() -> None:
    env = _make_env()
    env.reset(seed=7)

    rng = np.random.default_rng(7)
    for _ in range(50):
        action = rng.uniform(-1, 1, env.action_space.shape).astype(np.float32)
        obs, reward, terminated, truncated, _ = env.step(action)
        assert np.isfinite(obs).all()
        if terminated or truncated:
            env.reset()


def test_env_terminate_on_fall() -> None:
    env = _make_env()
    env.reset(seed=1)

    # drop base to ground
    env.data.qpos[2] = 0.1  # below min_base_height=0.3
    terminated = env._is_terminated()
    assert terminated, "expected terminated=True at low height"


def test_action_space_bounds() -> None:
    env = _make_env()
    assert env.action_space.low.min() == -1.0
    assert env.action_space.high.max() == 1.0
    assert env.action_space.shape == (env.model.nu,)


def test_config_update_changes_command_range() -> None:
    env = _make_env()
    env.reset(seed=0)

    env.update_config(cmd_vx_range=(-0.3, 0.8), cmd_vy_range=(-0.3, 0.3))
    assert env.config.cmd_vx_range == (-0.3, 0.8)
    assert env.config.cmd_vy_range == (-0.3, 0.3)


def test_push_force_is_cleared_after_step() -> None:
    scene = (
        Path(__file__).resolve().parents[1]
        / "mujoco_menagerie"
        / "unitree_g1"
        / "scene.xml"
    )
    if not scene.exists():
        pytest.skip(f"G1 scene file not found: {scene}")
    env = G1LocomotionEnv(
        G1Config(
            mjcf_path=scene,
            push_enabled=True,
            push_interval_steps=1,
            push_force_range=10.0,
        )
    )
    env.reset(seed=0)

    action = np.zeros(env.action_space.shape, dtype=np.float32)
    env.step(action)

    assert np.allclose(env.data.xfrc_applied[env._base_body_id, :3], 0.0)
