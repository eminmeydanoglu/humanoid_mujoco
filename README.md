# humanoid-mujoco

MuJoCo + Unitree G1 teleop starter project.

## Setup

```bash
uv sync
git clone --depth 1 --filter=blob:none --sparse https://github.com/google-deepmind/mujoco_menagerie.git
git -C mujoco_menagerie sparse-checkout set unitree_g1
```

## Run G1 keyboard teleop

```bash
uv run g1-teleop
```

Controls:

- `W/S`: forward +/-
- `A/D`: lateral left/right
- `Q/E`: yaw left/right
- `U/J`: arm lift +/-
- `H/K`: arm swing left/right
- `SPACE`: zero all commands
- `R`: reset to stand pose

## Run tests

```bash
uv run pytest -q
```
