from __future__ import annotations

from dataclasses import dataclass


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


@dataclass(slots=True)
class TeleopCommand:
    forward: float = 0.0
    lateral: float = 0.0
    yaw: float = 0.0
    arm_lift: float = 0.0
    arm_swing: float = 0.0


class TeleopCommandAccumulator:
    """Stateful keyboard command accumulator (press-to-adjust)."""

    def __init__(self, step: float = 0.1):
        self.step = step
        self.command = TeleopCommand()
        self._reset_requested = False

    def apply_action(self, action: str) -> TeleopCommand:
        c = self.command
        s = self.step

        if action == "forward_inc":
            c.forward = _clip(c.forward + s, -1.5, 1.5)
        elif action == "forward_dec":
            c.forward = _clip(c.forward - s, -1.5, 1.5)
        elif action == "lateral_left":
            c.lateral = _clip(c.lateral + s, -1.5, 1.5)
        elif action == "lateral_right":
            c.lateral = _clip(c.lateral - s, -1.5, 1.5)
        elif action == "yaw_left":
            c.yaw = _clip(c.yaw + s, -1.5, 1.5)
        elif action == "yaw_right":
            c.yaw = _clip(c.yaw - s, -1.5, 1.5)
        elif action == "arm_lift_up":
            c.arm_lift = _clip(c.arm_lift + s, -1.0, 1.0)
        elif action == "arm_lift_down":
            c.arm_lift = _clip(c.arm_lift - s, -1.0, 1.0)
        elif action == "arm_swing_left":
            c.arm_swing = _clip(c.arm_swing + s, -1.0, 1.0)
        elif action == "arm_swing_right":
            c.arm_swing = _clip(c.arm_swing - s, -1.0, 1.0)
        elif action == "zero_all":
            self.command = TeleopCommand()
        elif action == "reset_pose":
            self._reset_requested = True

        return self.command

    def consume_reset_request(self) -> bool:
        requested = self._reset_requested
        self._reset_requested = False
        return requested

    def pretty(self) -> str:
        c = self.command
        return (
            f"cmd forward={c.forward:+.2f} lateral={c.lateral:+.2f} "
            f"yaw={c.yaw:+.2f} arm_lift={c.arm_lift:+.2f} "
            f"arm_swing={c.arm_swing:+.2f}"
        )
