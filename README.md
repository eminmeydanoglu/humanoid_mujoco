# Unitree G1 Locomotion — MuJoCo RL

Train a [Unitree G1](https://www.unitree.com/g1/) humanoid robot to track velocity commands using PPO in MuJoCo simulation, then visualize and teleoperate the learned policy.

## Overview

- **Algorithm**: PPO (stable-baselines3) with VecNormalize and 16 parallel environments
- **Simulator**: MuJoCo 3.x via [mujoco-menagerie](https://github.com/google-deepmind/mujoco_menagerie) MJCF model
- **Observation**: 102-dim — base linear/angular velocity, projected gravity, velocity command, joint positions/velocities, last action, gait clock
- **Action**: 23-dim normalized joint position targets (scaled to ±0.1 rad before adding to default pose)
- **Reward**: 13 terms covering velocity tracking, upright posture, gait timing, foot clearance, energy efficiency

## Requirements

- Python ≥ 3.11
- [uv](https://github.com/astral-sh/uv)
- `mujoco_menagerie/` submodule (run `git submodule update --init`)

## Installation

```bash
git submodule update --init
uv sync
```

## Usage

### Train from scratch

```bash
uv run python train.py --timesteps 50_000_000 --n-envs 16
```

### Resume from checkpoint

```bash
uv run python train.py \
  --load-path g1_policy_checkpoints/PPO_6/g1_policy_48000000_steps.zip \
  --curriculum-stage 3
```

### Visualize a trained policy

Interactive viewer with keyboard velocity commands:

```bash
uv run python play.py --policy g1_policy.zip
```

| Key | Action |
|-----|--------|
| W / S | +/− forward velocity |
| A / D | +/− lateral velocity |
| Q / E | +/− yaw rate |
| Space | Zero all commands |
| R | Reset episode |

### Headless batch evaluation

```bash
uv run python play.py --policy g1_policy.zip --headless-steps 2000
```

### Keyboard teleoperation (without RL policy)

Sinusoidal gait driven directly from keyboard commands:

```bash
g1-teleop
```

### Monitor training

```bash
uv run tensorboard --logdir logs/
```

### Run tests

```bash
uv run pytest tests/
```

## Curriculum

Training progresses through four stages, each widening command ranges and enabling disturbances. `CurriculumCallback` auto-advances when the rolling mean reward exceeds **5.0** for three consecutive evaluation windows (50 K-step intervals).

| Stage | vx (m/s) | vy (m/s) | yaw (rad/s) | Push forces |
|-------|----------|----------|-------------|-------------|
| 0 | 0.3 → 0.8 | 0.0 | 0.0 | off |
| 1 | −0.3 → 0.8 | ±0.2 | ±0.3 | off |
| 2 | −0.3 → 0.8 | ±0.3 | ±0.5 | off |
| 3 | −0.3 → 0.8 | ±0.3 | ±0.5 | **on** |

## Reward terms

| Term | Weight | Description |
|------|--------|-------------|
| `lin_vel` | 2.0 | xy velocity tracking (Gaussian) |
| `ang_vel` | 1.0 | yaw rate tracking (Gaussian) |
| `alive` | 0.5 | constant per-step bonus |
| `feet_contact` | 1.0 | foot contact timing vs. gait clock |
| `feet_clear` | 0.5 | foot clearance during swing phase |
| `orientation` | −0.2 | upright posture penalty |
| `base_height` | −0.1 | height deviation from 0.78 m |
| `lin_vel_z` | −0.5 | vertical velocity penalty |
| `ang_vel_xy` | −0.05 | roll/pitch rate penalty |
| `torques` | −0.0002 | actuator force penalty |
| `joint_vel` | −0.0001 | joint velocity penalty |
| `action_rate` | −0.005 | action smoothness penalty |
| `dof_limit` | −1.0 | soft joint limit penalty |

## Checkpoints

Checkpoints are saved in pairs — both files must be present for correct inference:

```
g1_policy_{N}_steps.zip          ← PPO policy weights
g1_policy_{N}_steps_vecnorm.pkl  ← VecNormalize statistics
```

`play.py` loads both automatically when given the `.zip` path.

## Project structure

```
humanoid_mujoco/
  config/g1_config.py       — G1Config dataclass + curriculum definitions
  envs/g1_env.py            — G1LocomotionEnv (gymnasium.Env)
  rewards/reward_functions.py — 13 reward terms + helper math
  teleop/                   — keyboard teleoperation (no RL policy)
train.py                    — PPO setup, callbacks, VecNormalize, resume logic
play.py                     — policy rollout with MuJoCo viewer
tests/                      — pytest suite
mujoco_menagerie/           — git submodule (MJCF models)
g1_policy_checkpoints/      — saved .zip + _vecnorm.pkl pairs
logs/                       — TensorBoard event files
```
