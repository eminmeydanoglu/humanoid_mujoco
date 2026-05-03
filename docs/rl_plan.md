# G1 Locomotion RL — Implementation Plan

Train a Unitree G1 robot to track velocity commands in MuJoCo simulation using PPO.
Goal: a stable locomotion policy that tracks `(vx, vy, yaw_rate)` commands.

---

## Architecture Decisions

| Topic | Decision | Rationale |
|---|---|---|
| Backend | Standard MuJoCo + gymnasium | No GPU needed, simple setup |
| Control | Position actuators | g1.xml unchanged, consistent with existing code |
| PPO | stable-baselines3 | Battle-tested, direct gymnasium integration |
| Parallel envs | `SubprocVecEnv` | CPU multiprocessing, 16–32 envs |
| Python | 3.11+ | Repo standard |

---

## File Structure

```
humanoid_mujoco/
├── config/
│   └── g1_config.py          # G1Config dataclass
├── envs/
│   ├── __init__.py
│   └── g1_env.py             # gymnasium.Env
├── rewards/
│   ├── __init__.py
│   └── reward_functions.py   # 13 reward terms
├── teleop/                    # existing — do not modify
└── __init__.py
train.py                       # PPO training entry point
play.py                        # trained policy visualisation
```

---

## Dependencies

```bash
uv add gymnasium numpy "stable-baselines3>=2.3.0" tensorboard rich tqdm
```

```toml
# pyproject.toml
dependencies = [
    "mujoco>=3.5.0",
    "gymnasium>=1.2.3",
    "numpy>=2.4.2",
    "stable-baselines3>=2.3.0",
    "tensorboard>=2.20.0",
    "rich>=15.0.0",
    "tqdm>=4.67.3",
]
```

---

## Module 1: Configuration — `humanoid_mujoco/config/g1_config.py`

```python
@dataclass(slots=True)
class G1Config:
    # Paths
    mjcf_path: Path = ...  # mujoco_menagerie/unitree_g1/scene.xml

    # Simulation
    dt: float = 0.02            # 50 Hz
    gait_period: float = 0.8    # seconds
    action_scale: float = 0.1   # position target scale (kp=500 → max ~50 N⋅m)

    # Episode
    max_episode_steps: int = 1000   # 20 s (50 Hz × 1000)

    # Velocity command ranges (stage 0 defaults; widened by curriculum)
    cmd_vx_range: tuple[float, float] = (0.3, 0.8)
    cmd_vy_range: tuple[float, float] = (0.0, 0.0)
    cmd_yaw_range: tuple[float, float] = (0.0, 0.0)

    # Termination thresholds
    min_base_height: float = 0.3
    max_roll: float = 0.785    # 45 deg in radians
    max_pitch: float = 0.785

    # Target base height
    target_base_height: float = 0.78  # metres

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

    # Domain randomization
    friction_range: tuple[float, float] = (0.3, 1.5)
    mass_offset_range: tuple[float, float] = (-2.0, 2.0)
    push_enabled: bool = False
    push_interval_steps: int = 300
    push_force_range: float = 50.0  # Newtons
```

---

## Module 2: Reward Functions — `humanoid_mujoco/rewards/reward_functions.py`

| # | Function | Weight | Formula |
|---|---|---|---|
| 1 | `reward_lin_vel_tracking` | +2.0 | exp(−‖cmd_xy − vel_xy‖² / σ²) |
| 2 | `reward_ang_vel_tracking` | +1.0 | exp(−(cmd_yaw − ang_vel_z)² / σ²) |
| 3 | alive bonus | +0.5 | constant |
| 4 | `reward_feet_contact_timing` | +1.0 | foot contact synced to gait clock |
| 5 | `reward_feet_clearance` | +0.5 | foot height during swing phase |
| 6 | `penalty_orientation` | −0.2 | gravity_x² + gravity_y² |
| 7 | `penalty_base_height` | −0.1 | (base_z − 0.78)² |
| 8 | `penalty_lin_vel_z` | −0.5 | base_vel_z² |
| 9 | `penalty_ang_vel_xy` | −0.05 | ang_vel_x² + ang_vel_y² |
| 10 | `penalty_torques` | −2e-4 | Σ(actuator_force²) |
| 11 | `penalty_joint_vel` | −1e-4 | Σ(qvel[6:]²) |
| 12 | `penalty_action_rate` | −0.005 | Σ((action − last_action)²) |
| 13 | `penalty_soft_dof_pos_limit` | −1.0 | Σ max(0, \|normalized\| − 0.9)² |

`compute_reward_terms()` returns a dict (logged to TensorBoard per rollout).
`compute_reward()` returns the weighted sum.

---

## Module 3: Gymnasium Environment — `humanoid_mujoco/envs/g1_env.py`

### Observation Vector

| Field | Dim | Source |
|---|---|---|
| `base_lin_vel` | 3 | `data.qvel[0:3]` → body frame |
| `base_ang_vel` | 3 | `data.qvel[3:6]` → body frame |
| `projected_gravity` | 3 | gravity rotated to body frame |
| `velocity_command` | 3 | randomly sampled target |
| `joint_positions` | `nv−6` | `qpos[7:] − default_qpos[7:]` |
| `joint_velocities` | `nv−6` | `qvel[6:]` |
| `last_action` | `nu` | previous step ctrl |
| `clock_signal` | 2 | `[sin(phase), cos(phase)]` |

**Total:** `14 + 2*(nv−6) + nu` dims (~102 for G1)

### `step(action)`

```python
ctrl = base_ctrl + config.action_scale * action
ctrl = np.clip(ctrl, model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1])
data.ctrl[:] = ctrl
mujoco.mj_step(model, data)
phase = (phase + 2 * pi * dt / gait_period) % (2 * pi)
```

### Termination Conditions

```python
terminated = (
    base_height < config.min_base_height
    or abs(roll)  > config.max_roll
    or abs(pitch) > config.max_pitch
)
truncated = (step_count >= config.max_episode_steps)
```

### Domain Randomization (every `reset()`)

```python
model.geom_friction[:, 0] *= rng.uniform(*config.friction_range)
model.body_mass[base_id]  += rng.uniform(*config.mass_offset_range)
data.qpos += rng.normal(0, 0.02, data.qpos.shape)
data.qvel += rng.normal(0, 0.1,  data.qvel.shape)
```

Push disturbance applied every `push_interval_steps` inside `step()` when `push_enabled=True`.

---

## Module 4: Training — `train.py`

```python
n_steps = 512
base_env = VecMonitor(SubprocVecEnv([make_env(config)] * n_envs))
env = VecNormalize(base_env, norm_obs=True, norm_reward=False, clip_obs=10.0, gamma=0.99)

model = PPO(
    "MlpPolicy",
    env,
    policy_kwargs=dict(net_arch=[512, 256, 128], activation_fn=nn.ELU, log_std_init=-1.0),
    learning_rate=lambda f: 3e-4 * f,   # linear decay
    n_steps=n_steps,                     # covers a full gait cycle
    batch_size=n_envs * n_steps // 4,   # 4 minibatches
    n_epochs=5,
    clip_range=0.2,
    clip_range_vf=0.2,
    gamma=0.99,
    gae_lambda=0.95,
    ent_coef=0.05,
    target_kl=0.01,
    vf_coef=1.0,
    max_grad_norm=1.0,
    tensorboard_log="./logs/",
)
model.learn(total_timesteps=50_000_000, callback=CallbackList([...]))
```

### Callbacks

| Callback | Role |
|---|---|
| `CheckpointCallback` | saves `.zip` every 1M steps |
| `VecNormalizeSaveCallback` | saves `_vecnorm.pkl` alongside each checkpoint |
| `CurriculumCallback` | advances stage after 3 consecutive 50k-step windows above threshold |
| `RewardTermLogger` | logs all 13 reward terms to TensorBoard each rollout |

### Curriculum (4 Stages)

| Stage | vx (m/s) | vy (m/s) | yaw (rad/s) | Push |
|---|---|---|---|---|
| 0 | 0.3 → 0.8 | 0.0 | 0.0 | off |
| 1 | −0.3 → 0.8 | ±0.2 | ±0.3 | off |
| 2 | −0.3 → 0.8 | ±0.3 | ±0.5 | off |
| 3 | −0.3 → 0.8 | ±0.3 | ±0.5 | **on** |

Advance condition: `ep_rew_mean > 5.0` for 3 consecutive 50k-step evaluation windows.

---

## Module 5: Visualisation — `play.py`

Reuses the GLFW viewer loop and key bindings from `g1_keyboard.py`.
PPO policy `predict()` replaces `G1TeleopController.compute_ctrl()`.

```bash
uv run python play.py --policy g1_policy.zip
uv run python play.py --policy g1_policy.zip --headless-steps 500
```

---

## CLI Scripts (`pyproject.toml`)

```toml
[project.scripts]
g1-teleop = "humanoid_mujoco.teleop.g1_keyboard:main"
g1-train  = "train:main"
g1-play   = "play:main"
```

---

## Verification

```bash
# Dependencies
uv run python -c "import gymnasium; import stable_baselines3"

# Environment tests
uv run pytest tests/test_g1_env.py

# Short training run (~seconds)
uv run python train.py --timesteps 10000 --n-envs 4

# Policy smoke test
uv run python play.py --policy g1_policy.zip --headless-steps 500

# Full test suite
uv run pytest
```

### `tests/test_g1_env.py` — Test Coverage

- `test_env_reset_returns_valid_obs` — obs shape and dtype
- `test_observation_space_matches_obs` — gymnasium space bounds
- `test_env_step_zero_action` — step with zero action, results finite
- `test_env_step_random_actions` — 50 random-action steps
- `test_env_terminate_on_fall` — manipulate `qpos` and assert termination
- `test_action_space_bounds` — action space is `[-1, 1]` with correct shape
- `test_config_update_changes_command_range` — `update_config()` for curriculum
