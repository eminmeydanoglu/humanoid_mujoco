from __future__ import annotations

import argparse
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
from stable_baselines3 import PPO

from humanoid_mujoco.config.g1_config import G1Config
from humanoid_mujoco.envs.g1_env import G1LocomotionEnv


# Same key bindings as g1_keyboard.py
def _key_to_action() -> dict[int, str]:
    import glfw
    return {
        glfw.KEY_W: "vx_inc",
        glfw.KEY_S: "vx_dec",
        glfw.KEY_A: "vy_inc",
        glfw.KEY_D: "vy_dec",
        glfw.KEY_Q: "yaw_inc",
        glfw.KEY_E: "yaw_dec",
        glfw.KEY_SPACE: "zero",
        glfw.KEY_R: "reset",
    }


def _print_controls() -> None:
    print("G1 Policy Play")
    print("  W/S   vx (forward/back) +/-")
    print("  A/D   vy (left/right) +/-")
    print("  Q/E   yaw +/-")
    print("  SPACE zero commands")
    print("  R     reset pose")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualise a trained G1 policy")
    p.add_argument("--policy", type=Path, required=True, help="Path to policy .zip file")
    p.add_argument("--headless-steps", type=int, default=0, help="Run N steps without viewer and exit")
    p.add_argument("--realtime-scale", type=float, default=1.0)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    config = G1Config()
    if not config.mjcf_path.exists():
        raise FileNotFoundError(f"MJCF not found: {config.mjcf_path}")

    env = G1LocomotionEnv(config)
    policy = PPO.load(str(args.policy), device="cpu")

    _print_controls()

    if args.headless_steps > 0:
        obs, _ = env.reset()
        total_reward = 0.0
        for step in range(args.headless_steps):
            action, _ = policy.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            if terminated or truncated:
                obs, _ = env.reset()
        print(f"headless_done steps={args.headless_steps} total_reward={total_reward:.2f}")
        return

    # Keyboard command state
    cmd = np.zeros(3, dtype=np.float64)  # [vx, vy, yaw]
    step_size = 0.1
    do_reset = False
    key_map = _key_to_action()

    def on_key(keycode: int) -> None:
        nonlocal do_reset
        action = key_map.get(keycode)
        if action == "vx_inc":
            cmd[0] = min(cmd[0] + step_size, 1.5)
        elif action == "vx_dec":
            cmd[0] = max(cmd[0] - step_size, -1.5)
        elif action == "vy_inc":
            cmd[1] = min(cmd[1] + step_size, 1.5)
        elif action == "vy_dec":
            cmd[1] = max(cmd[1] - step_size, -1.5)
        elif action == "yaw_inc":
            cmd[2] = min(cmd[2] + step_size, 1.5)
        elif action == "yaw_dec":
            cmd[2] = max(cmd[2] - step_size, -1.5)
        elif action == "zero":
            cmd[:] = 0.0
        elif action == "reset":
            do_reset = True
        print(f"cmd vx={cmd[0]:+.2f} vy={cmd[1]:+.2f} yaw={cmd[2]:+.2f}")

    obs, _ = env.reset()
    model = env.model
    data = env.data

    with mujoco.viewer.launch_passive(model, data, key_callback=on_key) as viewer:
        while viewer.is_running():
            loop_start = time.perf_counter()

            if do_reset:
                obs, _ = env.reset()
                cmd[:] = 0.0
                do_reset = False

            # Propagate current command into env
            env._cmd[:] = cmd

            action, _ = policy.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(action)

            if terminated or truncated:
                obs, _ = env.reset()

            viewer.sync()

            if args.realtime_scale > 0.0:
                target_dt = model.opt.timestep / args.realtime_scale
                elapsed = time.perf_counter() - loop_start
                if elapsed < target_dt:
                    time.sleep(target_dt - elapsed)


if __name__ == "__main__":
    main()
