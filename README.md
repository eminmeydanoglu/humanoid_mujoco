# Unitree G1 Locomotion — MuJoCo RL

Train a [Unitree G1](https://www.unitree.com/g1/) humanoid robot to track velocity commands using PPO in MuJoCo simulation, then visualize and teleoperate the learned policy.

## Overview

- **Algorithm**: PPO (stable-baselines3) with VecNormalize and 16 parallel environments
- **Simulator**: MuJoCo 3.x via [mujoco-menagerie](https://github.com/google-deepmind/mujoco_menagerie) MJCF model
- **Observation**: 101-dim — base linear/angular velocity, projected gravity, velocity command, joint positions/velocities, last action, gait clock
- **Action**: 29-dim normalized joint position targets (scaled to ±0.5 rad before adding to default pose)
- **Reward**: 15 terms covering velocity tracking, upright posture, gait timing, foot clearance, energy efficiency, waist/ankle regularization

## Requirements

- Python ≥ 3.11
- [uv](https://github.com/astral-sh/uv)
- `mujoco_menagerie/` checkout containing `unitree_g1`

## Installation

```bash
uv sync
git clone --depth 1 --filter=blob:none --sparse https://github.com/google-deepmind/mujoco_menagerie.git
git -C mujoco_menagerie sparse-checkout set unitree_g1
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


## Reward terms

| Term | Weight | Description |
|------|--------|-------------|
| `lin_vel` | 2.0 | xy velocity tracking (Gaussian) |
| `ang_vel` | 1.0 | yaw rate tracking (Gaussian) |
| `alive` | 0.5 | constant per-step bonus |
| `feet_contact` | 2.0 | foot contact timing vs. gait clock |
| `feet_clear` | 1.5 | foot clearance during swing phase |
| `orientation` | −0.2 | upright posture penalty |
| `base_height` | −0.1 | height deviation from 0.78 m |
| `lin_vel_z` | −2.0 | vertical velocity penalty |
| `ang_vel_xy` | −0.05 | roll/pitch rate penalty |
| `torques` | −0.0002 | actuator force penalty |
| `joint_vel` | −0.0001 | joint velocity penalty |
| `action_rate` | −0.005 | action smoothness penalty |
| `dof_limit` | −1.0 | soft joint limit penalty |
| `waist_deviation` | −2.0 | waist joint deviation penalty |
| `ankle_deviation` | −2.0 | ankle pitch deviation penalty |

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
  rewards/reward_functions.py — 15 reward terms + helper math
  teleop/                   — keyboard teleoperation (no RL policy)
train.py                    — PPO setup, callbacks, VecNormalize, resume logic
play.py                     — policy rollout with MuJoCo viewer
tests/                      — pytest suite
mujoco_menagerie/           — sparse checkout of MJCF models
g1_policy_checkpoints/      — saved .zip + _vecnorm.pkl pairs
logs/                       — TensorBoard event files
```
