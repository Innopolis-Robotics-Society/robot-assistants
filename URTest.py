import math
import sys
import json
import time
import os
from pathlib import Path

import URBasic
import URBasic.robotModel
import URBasic.urScriptExt

from localizer import measure_xy_once as measure_xy


class ControlledRobot:
    """Класс для управления роботом."""

    VPATH = Path.home() / "take" / "ur10-assistant" / "hoba"
    PATH_STR = str(VPATH)

    RAW = 2.857
    PITCH = -1.309
    YAW = 0.0
    HOST = "192.168.0.100"

    INIT_POSE = [-0.23, -1, 0.7, RAW, PITCH, YAW]
    INTERMEDIATE_POSE = [-0.23, -1, 0.5, RAW, PITCH, YAW]

    def __init__(self):
        self.robot_model = URBasic.robotModel.RobotModel()

    # ---------------------------------------------------------------------

    @staticmethod
    def normalize_angle(angle):
        """Нормализует угол в диапазон [-pi, pi]."""
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle

    # ---------------------------------------------------------------------

    def check_position(
        self,
        current_pos,
        target_pos,
        linear_tolerance=0.001,
        angle_tolerance=0.01,
    ):
        """Проверяет, достигнута ли целевая позиция."""
        position_reached = True

        print("Проверка позиции:")

        for i, (current, target) in enumerate(zip(current_pos, target_pos)):

            if i < 3:
                # XYZ
                diff = abs(current - target)
                ok = diff <= linear_tolerance
                position_reached &= ok

                print(
                    f"Ось {i}(лин.): тек={current:08.6f}, "
                    f"цел={target:08.6f}, разн={diff:08.6f} {'✓' if ok else '✗'}"
                )
            else:
                # RX RY RZ
                norm_cur = self.normalize_angle(current)
                norm_tgt = self.normalize_angle(target)
                diff = abs(norm_cur - norm_tgt)
                ok = diff <= angle_tolerance
                position_reached &= ok

                print(
                    f"Ось {i}(угл.): тек={current:08.6f}→{norm_cur:08.6f}, "
                    f"цел={target:08.6f}→{norm_tgt:08.6f}, "
                    f"разн={diff:08.6f} {'✓' if ok else '✗'}"
                )

        print(f"Позиция достигнута: {position_reached}")
        print("-" * 80)

        return position_reached

    # ---------------------------------------------------------------------

    def wait_for_position(self, robot, target_pos, name="", timeout=15.0, interval=0.05):
        """Ожидает достижения позиции с таймаутом."""
        start = time.time()
        attempt = 0

        while time.time() - start < timeout:
            attempt += 1
            cur = robot.get_actual_tcp_pose_custom()

            print(f"Попытка {attempt} для {name}:")
            if self.check_position(cur, target_pos):
                print(f"✓ Позиция '{name}' достигнута!")
                return True

            if attempt % 10 == 0:
                print(f"Прогресс: {time.time() - start:.1f}s из {timeout}s")

            time.sleep(interval)

        print(f"✗ Таймаут: '{name}' не достигнута")
        return False

    # ---------------------------------------------------------------------

    @staticmethod
    def pause(name, sec=2.0):
        print(f"⏸️  Пауза {sec}s в позиции '{name}'...")
        time.sleep(sec)
        print("▶️  Продолжаем.")

    # ---------------------------------------------------------------------

    def move_to(self, robot, target_pos, name):
        """Унифицированный метод движения."""
        print(f"Переход: {name} → {target_pos}")
        robot.set_realtime_pose(target_pos)

        if not self.wait_for_position(robot, target_pos, name):
            robot.close()
            raise RuntimeError(f"Не удалось достичь: {name}")

    # ---------------------------------------------------------------------

    def measure_xy_camera(self):
        """Надёжное измерение координат камеры."""
        for attempt in range(5):
            res = measure_xy(display=False, warmup_s=1.5)
            if res is not None:
                return res
            time.sleep(0.2)

        return measure_xy(display=True, warmup_s=1.5)

    # ---------------------------------------------------------------------

    def run(self):
        assert self.VPATH.exists(), f"bad path: {self.VPATH}"

        if self.PATH_STR not in sys.path:
            sys.path.insert(0, self.PATH_STR)

        print("Initialization UR")

        robot = URBasic.urScriptExt.UrScriptExt(
            host=self.HOST, robotModel=self.robot_model
        )
        robot.init_realtime_control()

        try:
            cur = robot.get_actual_tcp_pose_custom()
            print("Текущая позиция:")
            print("[{: .6f}, {: .6f}, {: .6f}, {: .6f}, {: .6f}, {: .6f}]".format(*cur))

            input("Enter → начать...")

            # Move 1: init pose
            self.move_to(robot, self.INIT_POSE, "Начальная позиция")

            # XY via camera
            print("Измеряю координаты COM...")
            xy = self.measure_xy_camera()

            if xy is None:
                x = float(input("x (m): "))
                y = float(input("y (m): "))
            else:
                x, y = xy
                print(f"COM: x={x:.4f}, y={y:.4f}")

            x_adj = x - 0.23
            y_adj = y - 1
            z_0 = 0.505
            z_adj = z_0 - 0.3

            # Move 2: intermediate
            self.move_to(robot, self.INTERMEDIATE_POSE, "Промежуточная позиция")
            self.pause("Промежуточная позиция")

            # Move 3: approach (z0)
            approach = [x_adj, y_adj, z_0, self.RAW, self.PITCH, self.YAW]
            self.move_to(robot, approach, "Точка подхода (z0)")
            self.pause("Точка подхода (z0)")

            # Move 4: final low
            target = [x_adj, y_adj, z_adj, self.RAW, self.PITCH, self.YAW]
            self.move_to(robot, target, "Нижняя точка")
            self.pause("Нижняя точка")

            # Return
            self.move_to(robot, self.INTERMEDIATE_POSE, "Промежуточная (возврат)")
            self.pause("Промежуточная (возврат)")

            print("Готово.")

        finally:
            robot.close()
