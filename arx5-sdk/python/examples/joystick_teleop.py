import time
import os
import sys

import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)
from arx5_interface import (
    Arx5CartesianController,
    EEFState,
    Gain,
    LogLevel,
)

from peripherals.joystick import JoystickRobotics, XboxButton

import time
import click


def start_joystick_teleop(controller: Arx5CartesianController, joystick: JoystickRobotics):

    cmd_dt = 0.01
    preview_time = 0.1
    print("Starting teleop control with pre-calibrated Joystick...")

    start_time = time.monotonic()
    loop_cnt = 0

    while True:
        eef_state = controller.get_eef_state()

        # Get joystick state (absolute pose from joystick)
        joystick_pose, gripper_pos, control_button = joystick.get_control()

        print(
            f"Current --- time: {time.monotonic() - start_time:.03f}s, x: {eef_state.pose_6d()[0]:.03f}, y: {eef_state.pose_6d()[1]:.03f}, z: {eef_state.pose_6d()[2]:.03f}, roll: {eef_state.pose_6d()[3]:.03f}, pitch: {eef_state.pose_6d()[4]:.03f}, yaw: {eef_state.pose_6d()[5]:.03f}, gripper: {eef_state.gripper_pos:.03f}    ",
            end="\r",
        )

        # Check for exit command
        if control_button == XboxButton.X or control_button == XboxButton.B:
            print("\nExit buttons pressed. Resetting to home and exiting...")
            break  # Exit the control loop

        current_timestamp = controller.get_timestamp()
        eef_cmd = EEFState()
        eef_cmd.pose_6d()[:] = joystick_pose
        eef_cmd.gripper_pos = gripper_pos
        eef_cmd.timestamp = current_timestamp + preview_time
        controller.set_eef_cmd(eef_cmd)

        # wait
        loop_cnt += 1
        target_time = start_time + loop_cnt * cmd_dt
        sleep_time = target_time - time.monotonic()
        if sleep_time > 0:
            time.sleep(sleep_time)


@click.command()
@click.argument("model")  # ARX arm model: X5 or L5
@click.argument("interface")  # can bus name (can0 etc.)
def main(model: str, interface: str):
    # Initialize robot controller
    print("Initializing robot controller...")
    controller = Arx5CartesianController(model, interface)
    controller.reset_to_home()

    robot_config = controller.get_robot_config()
    controller.set_log_level(LogLevel.DEBUG)

    # Initialize Joystick (includes 2s calibration)
    print("Initializing Joystick controller...")
    joystick = JoystickRobotics(
        home_position=controller.get_home_pose().tolist()[:3],
        ee_limit=[[0.0, -0.5, -0.5, -1.8, -1.6, -1.6], [0.7, 0.5, 0.5, 1.8, 1.6, 1.6]],
        gripper_limit=[0.0, robot_config.gripper_width],
    )
    print("Joystick calibration completed.")

    np.set_printoptions(precision=4, suppress=True)
    start_joystick_teleop(controller, joystick)
    controller.reset_to_home()
    controller.set_to_damping()


if __name__ == "__main__":
    main()
