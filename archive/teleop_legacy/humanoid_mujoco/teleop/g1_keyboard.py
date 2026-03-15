from __future__ import annotations

import argparse
import time
from pathlib import Path

import mujoco
import mujoco.viewer

from humanoid_mujoco.teleop.commanding import TeleopCommandAccumulator
from humanoid_mujoco.teleop.g1_controller import G1TeleopController
from humanoid_mujoco.teleop.g1_controller import find_keyframe_id


def _default_mjcf() -> Path:
    root = Path(__file__).resolve().parents[2]
    return root / "mujoco_menagerie" / "unitree_g1" / "scene.xml"


def _key_to_action() -> dict[int, str]:
    import glfw

    return {
        glfw.KEY_W: "forward_inc",
        glfw.KEY_S: "forward_dec",
        glfw.KEY_A: "lateral_left",
        glfw.KEY_D: "lateral_right",
        glfw.KEY_Q: "yaw_left",
        glfw.KEY_E: "yaw_right",
        glfw.KEY_U: "arm_lift_up",
        glfw.KEY_J: "arm_lift_down",
        glfw.KEY_H: "arm_swing_left",
        glfw.KEY_K: "arm_swing_right",
        glfw.KEY_SPACE: "zero_all",
        glfw.KEY_R: "reset_pose",
    }


def _print_controls() -> None:
    print("G1 Teleop controls")
    print("  W/S forward +/-")
    print("  A/D lateral left/right")
    print("  Q/E yaw left/right")
    print("  U/J arm lift +/-")
    print("  H/K arm swing left/right")
    print("  SPACE zero commands")
    print("  R reset to stand keyframe")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Keyboard teleop for Unitree G1")
    parser.add_argument(
        "--mjcf",
        type=Path,
        default=_default_mjcf(),
        help="Path to MJCF/scene XML",
    )
    parser.add_argument(
        "--realtime-scale",
        type=float,
        default=1.0,
        help="1.0 = real-time, 0 = run as fast as possible",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=0,
        help="0 means infinite loop until viewer closes",
    )
    parser.add_argument(
        "--headless-steps",
        type=int,
        default=0,
        help="Run without viewer for N steps and exit",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.realtime_scale < 0.0:
        raise ValueError("--realtime-scale must be >= 0")
    if args.max_steps < 0:
        raise ValueError("--max-steps must be >= 0")
    if args.headless_steps < 0:
        raise ValueError("--headless-steps must be >= 0")

    if not args.mjcf.exists():
        raise FileNotFoundError(f"MJCF not found: {args.mjcf}")

    model = mujoco.MjModel.from_xml_path(args.mjcf.as_posix())
    data = mujoco.MjData(model)

    stand_id = find_keyframe_id(model, "stand")
    if stand_id is not None:
        mujoco.mj_resetDataKeyframe(model, data, stand_id)

    controller = G1TeleopController(model)
    accumulator = TeleopCommandAccumulator(step=0.1)
    key_map = _key_to_action()

    def on_key(keycode: int) -> None:
        action = key_map.get(keycode)
        if action is None:
            return
        accumulator.apply_action(action)
        print(accumulator.pretty())

    _print_controls()

    if args.headless_steps > 0:
        for _ in range(args.headless_steps):
            data.ctrl[:] = controller.compute_ctrl(accumulator.command, data.time)
            mujoco.mj_step(model, data)
        print(f"headless_done steps={args.headless_steps} time={data.time:.3f}")
        return

    step_count = 0
    with mujoco.viewer.launch_passive(model, data, key_callback=on_key) as viewer:
        while viewer.is_running():
            loop_start = time.perf_counter()

            if accumulator.consume_reset_request():
                if stand_id is not None:
                    mujoco.mj_resetDataKeyframe(model, data, stand_id)
                else:
                    mujoco.mj_resetData(model, data)
                accumulator.apply_action("zero_all")
                print(accumulator.pretty())

            data.ctrl[:] = controller.compute_ctrl(accumulator.command, data.time)
            mujoco.mj_step(model, data)
            viewer.sync()

            step_count += 1
            if args.max_steps > 0 and step_count >= args.max_steps:
                break

            if args.realtime_scale > 0.0:
                target_dt = model.opt.timestep / args.realtime_scale
                elapsed = time.perf_counter() - loop_start
                if elapsed < target_dt:
                    time.sleep(target_dt - elapsed)


if __name__ == "__main__":
    main()
