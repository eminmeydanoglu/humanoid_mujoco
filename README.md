# humanoid-mujoco

Unitree G1 MuJoCo workspace, now focused on RL-based locomotion.

Legacy keyboard teleop code is archived under `archive/teleop_legacy/`.

## Setup

```bash
uv sync
git clone --depth 1 --filter=blob:none --sparse https://github.com/google-deepmind/mujoco_menagerie.git
git -C mujoco_menagerie sparse-checkout set unitree_g1
```

## Next

RL execution plan: `docs/rl_plan.md`.
