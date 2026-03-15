from humanoid_mujoco.teleop.commanding import TeleopCommandAccumulator


def test_command_accumulator_clamps_forward() -> None:
    acc = TeleopCommandAccumulator(step=0.2)
    for _ in range(20):
        acc.apply_action("forward_inc")
    assert acc.command.forward == 1.5

    for _ in range(20):
        acc.apply_action("forward_dec")
    assert acc.command.forward == -1.5


def test_command_accumulator_zero_all() -> None:
    acc = TeleopCommandAccumulator(step=0.1)
    acc.apply_action("forward_inc")
    acc.apply_action("yaw_left")
    acc.apply_action("arm_lift_up")
    acc.apply_action("zero_all")
    assert acc.command.forward == 0.0
    assert acc.command.yaw == 0.0
    assert acc.command.arm_lift == 0.0


def test_command_accumulator_reset_flag_consumed_once() -> None:
    acc = TeleopCommandAccumulator(step=0.1)
    acc.apply_action("reset_pose")
    assert acc.consume_reset_request() is True
    assert acc.consume_reset_request() is False


def test_reset_flow_can_clear_existing_command() -> None:
    acc = TeleopCommandAccumulator(step=0.1)
    acc.apply_action("forward_inc")
    acc.apply_action("yaw_left")
    acc.apply_action("reset_pose")
    assert acc.consume_reset_request() is True
    acc.apply_action("zero_all")
    assert acc.command.forward == 0.0
    assert acc.command.yaw == 0.0
