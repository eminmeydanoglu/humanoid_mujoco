# RL Plan for G1 Locomotion (MuJoCo)

## Objective

Train a policy that tracks velocity commands (`vx`, `vy`, `wz`) and produces stable walking in simulation, then prepare for transfer to real robot.

## Success Criteria

- Tracks command velocity with low steady-state error.
- No frequent falls on flat terrain for long episodes.
- Recovers from moderate pushes.
- Exports a deterministic inference policy with fixed control rate.

## Phase 0 - Environment Foundation

1. Build `G1LocomotionEnv` around MuJoCo.
2. Define control loop rates:
   - physics: 500-1000 Hz
   - policy: 50-100 Hz
3. Add reset logic with randomized initial states around standing keyframe.
4. Add termination conditions (fall, joint limit violation, NaN/Inf).

Deliverable: reproducible env step/reset API and rollout script.

## Phase 1 - Observation, Action, Reward Design

### Observation

- Base orientation (gravity projection or quaternion), base angular velocity.
- Joint positions/velocities (optionally relative to nominal pose).
- Previous action.
- Command input (`vx`, `vy`, `wz`).

### Action

- Joint target offsets for selected joints, then map to actuator targets.
- Action clipping and rate limiting.

### Reward

- + command tracking (`vx`, `vy`, `wz`).
- + alive/upright bonus.
- - energy / action magnitude / action rate.
- - foot slip and large impact penalties.
- - posture deviation penalty.

Deliverable: reward breakdown logger with per-term plots.

## Phase 2 - PPO Baseline

1. Implement PPO training pipeline (vectorized env).
2. Start on flat terrain only.
3. Add curriculum: slow to fast commands, wider command range over time.
4. Save checkpoints and eval videos periodically.

Suggested defaults:

- horizon: 24-32 policy steps
- gamma: 0.99
- gae_lambda: 0.95
- clip: 0.2
- entropy coef: small non-zero

Deliverable: policy that walks stably and tracks commanded velocity on flat ground.

## Phase 3 - Robustness & Domain Randomization

1. Randomize mass/friction/damping within bounded ranges.
2. Add control latency and observation noise.
3. Add random pushes.
4. Add terrain variants (mild slopes, small height perturbations).

Deliverable: robust policy with reduced performance drop under perturbations.

## Phase 4 - Sim2Real Preparation

1. Match command and sensing interfaces to robot runtime API.
2. Enforce real actuator limits, rate limits, and watchdog behavior.
3. Run system identification loop using real logs (joint tracking, latency, IMU stats).
4. Freeze deployment graph for deterministic inference.

Deliverable: deployment-ready policy package and safety checklist.

## Tooling and Repo Tasks

1. Create `humanoid_mujoco/rl/env.py` and `humanoid_mujoco/rl/train.py`.
2. Add config system (`configs/locomotion_ppo.yaml`).
3. Add logging (`tensorboard` or `wandb`) and eval scripts.
4. Add regression tests for env reset/step numerical stability.

## Milestones

- M1: Environment + reward debug complete.
- M2: Flat-ground PPO walk/tracking baseline.
- M3: Robust policy with randomization.
- M4: Sim2real handoff package.
