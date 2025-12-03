import threading
import time
from typing import List, Tuple, Dict
from enum import Enum

import pygame


class XboxButton(Enum):
    A = "A"
    B = "B"
    X = "X"
    Y = "Y"
    LB = "LB"
    RB = "RB"
    VIEW = "VIEW"
    MENU = "MENU"
    NONE = "NONE"


class JoystickRobotics:
    def __init__(
        self,
        trans_step: List[float] = [0.001, 0.001, 0.001],
        orient_step: List[float] = [0.01, 0.005, 0.005],  # roll, pitch, yaw（rad/step）
        gripper_step: float = 0.001,
        trans_reverse: List[int] = [1, 1, 1],
        euler_reverse: List[int] = [1, 1, 1],
        home_position: List[float] = [0.0, 0.0, 0.0],
        home_euler: List[float] = [0.0, 0.0, 0.0],
        home_gripper: float = 0.0,
        ee_limit: List[List[float]] = [[-10, -10, -10, -10, -10, -10], [10, 10, 10, 10, 10, 10]],
        gripper_limit: List[float] = [0.0, 1.0],
        loop_period: float = 0.01,
        idx: int = 0,  # index of joystick
    ) -> None:
        # 初始化 pygame 手柄
        pygame.init()
        pygame.joystick.init()
        if pygame.joystick.get_count() <= idx:
            raise RuntimeError("未找到手柄，或索引超出范围")
        self._js = pygame.joystick.Joystick(idx)
        self._js.init()

        # 状态
        self.position = list(home_position)
        self.euler = list(home_euler)  # roll, pitch, yaw
        self.gripper = home_gripper
        self.control_button = XboxButton.NONE

        # 参数
        self.trans_step = list(trans_step)
        self.orient_step = list(orient_step)
        self.gripper_step = float(gripper_step)
        self.gripper_limit = list(gripper_limit)
        self.trans_reverse = list(trans_reverse)
        self.euler_reverse = list(euler_reverse)
        self.ee_limit = [list(ee_limit[0]), list(ee_limit[1])]
        self.loop_period = float(loop_period)

        # home 位姿
        self.home_position = list(home_position)
        self.home_euler = list(home_euler)

        # 并发
        self._lock = threading.Lock()
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _apply_deadzone(self, v: float, dz: float) -> float:
        return 0.0 if abs(v) < dz else v

    # 获取摇杆轴
    def _axes(self) -> List[float]:
        l_stick_h_raw = self._js.get_axis(0)
        l_stick_v_raw = self._js.get_axis(1)
        r_stick_h_raw = self._js.get_axis(3)
        r_stick_v_raw = self._js.get_axis(4)
        l_stick_h = self._apply_deadzone(l_stick_h_raw, 0.2)
        l_stick_v = self._apply_deadzone(l_stick_v_raw, 0.2)
        r_stick_h = self._apply_deadzone(r_stick_h_raw, 0.2)
        r_stick_v = self._apply_deadzone(r_stick_v_raw, 0.2)
        return [l_stick_h, l_stick_v, r_stick_h, r_stick_v]

    # 获取按钮状态
    def _buttons(self) -> Dict[XboxButton, int]:
        return {
            XboxButton.A: self._js.get_button(0),
            XboxButton.B: self._js.get_button(1),
            XboxButton.X: self._js.get_button(2),
            XboxButton.Y: self._js.get_button(3),
            XboxButton.LB: self._js.get_button(4),
            XboxButton.RB: self._js.get_button(5),
            XboxButton.VIEW: self._js.get_button(6),
            XboxButton.MENU: self._js.get_button(7),
        }

    # 获取十字键状态
    def _hats(self) -> Tuple[float, float]:
        return self._js.get_hat(0)

    def _clamp_pose(self) -> None:
        # 位置 xyz
        for i in range(3):
            self.position[i] = max(self.ee_limit[0][i], min(self.ee_limit[1][i], self.position[i]))
        # 姿态 rpy
        for i in range(3):
            j = 3 + i
            self.euler[i] = max(self.ee_limit[0][j], min(self.ee_limit[1][j], self.euler[i]))
        # 夹爪
        self.gripper = max(self.gripper_limit[0], min(self.gripper_limit[1], self.gripper))

    def _loop(self) -> None:
        while self._running:
            try:
                pygame.event.pump()
                l_stick_h, l_stick_v, r_stick_h, r_stick_v = self._axes()
                buttons = self._buttons()
                hat_h, hat_v = self._hats()

                with self._lock:
                    # 平移（左摇杆，Y 上，A 下）
                    self.position[0] += self.trans_step[0] * (-l_stick_v) * self.trans_reverse[0]
                    self.position[1] += self.trans_step[1] * (-l_stick_h) * self.trans_reverse[1]
                    self.position[2] += self.trans_step[2] * (hat_v) * self.trans_reverse[2]

                    # 姿态（右摇杆：水平=roll，垂直=pitch；X/B 调 yaw）
                    self.euler[0] += self.orient_step[0] * (hat_h) * self.euler_reverse[0]  # roll
                    self.euler[1] += self.orient_step[1] * (-r_stick_v) * self.euler_reverse[1]  # pitch
                    self.euler[2] += self.orient_step[2] * (-r_stick_h) * self.euler_reverse[2]  # yaw

                    # 夹爪（LB 关，RB 开）
                    if buttons[XboxButton.LB]:
                        self.gripper -= self.gripper_step
                    if buttons[XboxButton.RB]:
                        self.gripper += self.gripper_step

                    self._clamp_pose()

                    # 按住 View 键，缓慢回到 home 位置
                    if buttons[XboxButton.VIEW]:
                        self._go_to_home()

                    # control button
                    if buttons[XboxButton.X]:
                        self.control_button = XboxButton.X
                    elif buttons[XboxButton.B]:
                        self.control_button = XboxButton.B
                    else:
                        self.control_button = XboxButton.NONE

                time.sleep(self.loop_period)
            except Exception:
                time.sleep(0.1)

    def _go_to_home(self):
        """缓慢将位姿回到 home（位置 + 姿态）。"""
        eps = 0.002
        step_scale = 1.0  # 回家速度倍率
        # 位置 x
        if self.position[0] > self.home_position[0] + eps:
            self.position[0] -= self.trans_step[0] * step_scale
        elif self.position[0] < self.home_position[0] - eps:
            self.position[0] += self.trans_step[0] * step_scale
        # 位置 y
        if self.position[1] > self.home_position[1] + eps:
            self.position[1] -= self.trans_step[1] * step_scale
        elif self.position[1] < self.home_position[1] - eps:
            self.position[1] += self.trans_step[1] * step_scale
        # 位置 z
        if self.position[2] > self.home_position[2] + eps:
            self.position[2] -= self.trans_step[2] * step_scale
        elif self.position[2] < self.home_position[2] - eps:
            self.position[2] += self.trans_step[2] * step_scale

        # 姿态 roll, pitch, yaw（弧度）
        eps_ang = 0.01  # 约 0.6°
        if self.euler[0] > self.home_euler[0] + eps_ang:
            self.euler[0] -= self.orient_step[0] * step_scale
        elif self.euler[0] < self.home_euler[0] - eps_ang:
            self.euler[0] += self.orient_step[0] * step_scale

        if self.euler[1] > self.home_euler[1] + eps_ang:
            self.euler[1] -= self.orient_step[1] * step_scale
        elif self.euler[1] < self.home_euler[1] - eps_ang:
            self.euler[1] += self.orient_step[1] * step_scale

        if self.euler[2] > self.home_euler[2] + eps_ang:
            self.euler[2] -= self.orient_step[2] * step_scale
        elif self.euler[2] < self.home_euler[2] - eps_ang:
            self.euler[2] += self.orient_step[2] * step_scale
        # 限幅
        self._clamp_pose()

    def get_control(self):
        with self._lock:
            x, y, z = self.position
            roll, pitch, yaw = self.euler
            posture = [x, y, z, roll, pitch, yaw]
            return posture, float(self.gripper), XboxButton(self.control_button)

    def stop(self) -> None:
        self._running = False
        if hasattr(self, "_thread"):
            self._thread.join(timeout=1.0)


if __name__ == "__main__":
    js = JoystickRobotics(
        ee_limit=[[0.0, -0.5, -0.5, -1.3, -1.3, -1.3], [0.5, 0.5, 0.5, 1.3, 1.3, 1.3]],
    )

    try:
        while True:
            pose, gripper, control_button = js.get_control()
            pose_str = ", ".join(f"{v:.2f}" for v in pose)
            print(f"pose=[{pose_str}], gripper={gripper:.2f}, control_button={control_button}")
            time.sleep(0.05)
    except KeyboardInterrupt:
        pass
    finally:
        js.stop()
