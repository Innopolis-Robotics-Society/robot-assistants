import math
import time
from typing import List

import rclpy
from rclpy.node import Node

import URBasic
import URBasic.robotModel
import URBasic.urScriptExt

from ur10e_ros2.srv import tcp_pose, egp_state 


def egp_close(r):
    # DO0 = 1 (Close), DO1 = 0
    r.set_standard_digital_out(0, True)
    r.set_standard_digital_out(1, False)


def egp_open(r):
    # DO0 = 0, DO1 = 1 (Open)
    r.set_standard_digital_out(0, False)
    r.set_standard_digital_out(1, True)


def egp_free(r):
    # DO0 = 0, DO1 = 0 (free / отключён)
    r.set_standard_digital_out(0, False)
    r.set_standard_digital_out(1, False)


def egp_brake(r):
    # DO0 = 1, DO1 = 1 (brake / удержание)
    r.set_standard_digital_out(0, True)
    r.set_standard_digital_out(1, True)


def normalize_angle(angle: float) -> float:
    """Нормализует угол в диапазон [-pi, pi]."""
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def check_position(
    current_pos: List[float],
    target_pos: List[float],
    linear_tolerance: float = 0.001,
    angle_tolerance: float = 0.05,
) -> bool:
    """
    Проверяет, достигнута ли целевая позиция TCP.
    current_pos / target_pos: [x, y, z, rx, ry, rz]
    linear_tolerance: допуск по метрам
    angle_tolerance: допуск по радианам
    """
    position_reached = True

    for i, (current, target) in enumerate(zip(current_pos, target_pos)):
        if i < 3:
            diff = abs(current - target)
            if diff > linear_tolerance:
                position_reached = False
        else:
            norm_current = normalize_angle(current)
            norm_target = normalize_angle(target)
            diff = abs(norm_current - norm_target)
            if diff > angle_tolerance:
                position_reached = False

    return position_reached


def wait_for_position(
    robot,
    target_pos: List[float],
    timeout: float = 15.0,
    check_interval: float = 0.05,
    linear_tolerance: float = 0.001,
    angle_tolerance: float = 0.05,
    node: Node | None = None,
) -> bool:
    """Ожидает достижения целевой позиции TCP с таймаутом."""
    start_time = time.time()
    attempt = 0

    while time.time() - start_time < timeout:
        attempt += 1

        try:
            current_pos = robot.get_actual_tcp_pose_custom()
        except Exception as e:
            if node:
                node.get_logger().error(f"Ошибка чтения текущей позы TCP: {e}")
            return False

        if check_position(current_pos, target_pos, linear_tolerance, angle_tolerance):
            if node:
                node.get_logger().info("Целевая TCP-позиция достигнута.")
            return True

        if attempt % 20 == 0 and node:
            elapsed = time.time() - start_time
            node.get_logger().info(
                f"Ожидание позиции: прошло {elapsed:.1f} c / {timeout:.1f} c"
            )

        time.sleep(check_interval)

    if node:
        node.get_logger().warn("Таймаут ожидания достижения целевой позиции TCP.")
    return False


class UR10eNode(Node):
    def __init__(self):
        super().__init__("ur10e_node")

        # Параметры ноды
        self.declare_parameter("robot_ip", "192.168.0.100")
        self.declare_parameter("wait_timeout", 15.0)
        self.declare_parameter("linear_tolerance", 0.001)
        self.declare_parameter("angle_tolerance", 0.05)

        robot_ip = self.get_parameter("robot_ip").get_parameter_value().string_value
        self.wait_timeout = (
            self.get_parameter("wait_timeout").get_parameter_value().double_value
        )
        self.linear_tolerance = (
            self.get_parameter("linear_tolerance").get_parameter_value().double_value
        )
        self.angle_tolerance = (
            self.get_parameter("angle_tolerance").get_parameter_value().double_value
        )

        # Инициализация URBasic
        self.get_logger().info(f"Подключение к UR10e по IP {robot_ip} ...")

        try:
            self.robot_model = URBasic.robotModel.RobotModel()
            self.robot = URBasic.urScriptExt.UrScriptExt(
                host=robot_ip,
                robotModel=self.robot_model,
            )
            # Реалтайм-режим, как в твоём голосовом ассистенте
            self.robot.init_realtime_control()

            # Просто чтобы убедиться, что связь есть
            tcp = self.robot.get_actual_tcp_pose_custom()
            self.get_logger().info(
                "Робот подключен. Текущая TCP поза: "
                + "[{: 08.6f}, {: 08.6f}, {: 08.6f}, {: 08.6f}, {: 08.6f}, {: 08.6f}]".format(
                    *tcp
                )
            )
        except Exception as e:
            self.get_logger().error(f"Не удалось инициализировать URBasic: {e}")
            raise

        # Сервисы
        self.set_tcp_pose_srv = self.create_service(
            tcp_pose,
            "ur10e/set_tcp_pose",
            self.set_tcp_pose_cb,
        )

        self.set_gripper_state_srv = self.create_service(
            egp_state,
            "ur10e/set_gripper_state",
            self.set_gripper_state_cb,
        )

        self.get_logger().info("UR10eNode запущена.")

    # ---------- Service: задать TCP-позу ---------- #

    def set_tcp_pose_cb(self, req: tcp_pose.Request, res: tcp_pose.Response):
        """
        Задать абсолютную TCP-позу робота.
        Поза задаётся в базовой системе координат робота: [x, y, z, rx, ry, rz].
        """
        target = [req.x, req.y, req.z, req.rx, req.ry, req.rz]
        self.get_logger().info(
            "Получен запрос set_tcp_pose: "
            + "[{: 08.6f}, {: 08.6f}, {: 08.6f}, {: 08.6f}, {: 08.6f}, {: 08.6f}]".format(
                *target
            )
        )

        try:
            # В твоём коде ты используешь set_realtime_pose
            self.robot.set_realtime_pose(target)
        except Exception as e:
            msg = f"Ошибка отправки позы в URBasic: {e}"
            self.get_logger().error(msg)
            res.success = False
            res.message = msg
            return res

        # Ожидание достижения позиции
        ok = wait_for_position(
            self.robot,
            target,
            timeout=self.wait_timeout,
            check_interval=0.05,
            linear_tolerance=self.linear_tolerance,
            angle_tolerance=self.angle_tolerance,
            node=self,
        )

        if not ok:
            res.success = False
            res.message = "Таймаут ожидания достижения позиции."
        else:
            res.success = True
            res.message = "Позиция достигнута."

        return res

    # ---------- Service: состояние грипера ---------- #

    def set_gripper_state_cb(
        self, req: egp_state.Request, res: egp_state.Response
    ):
        """
        Управление состоянием грипера Schunk EGP через цифровые выходы UR:
        state: 'open' | 'close' | 'free' | 'brake'
        """
        state = req.state.strip().lower()
        self.get_logger().info(f"Запрос set_gripper_state: '{state}'")

        try:
            if state == "open":
                egp_open(self.robot)
            elif state == "close":
                egp_close(self.robot)
            elif state == "free":
                egp_free(self.robot)
            elif state == "brake":
                egp_brake(self.robot)
            else:
                msg = "Неизвестное состояние грипера. Ожидается: open/close/free/brake."
                self.get_logger().warn(msg)
                res.success = False
                res.message = msg
                return res

            # маленькая пауза для надёжности
            time.sleep(0.2)

            res.success = True
            res.message = f"Гриппер переведён в состояние '{state}'."
            return res

        except Exception as e:
            msg = f"Ошибка управления гриппером: {e}"
            self.get_logger().error(msg)
            res.success = False
            res.message = msg
            return res

    # ---------- Завершение работы ---------- #

    def destroy_node(self):
        # корректно отпускаем гриппер и закрываем соединение
        try:
            egp_free(self.robot)
        except Exception:
            pass

        try:
            self.robot.close()
        except Exception:
            pass

        self.get_logger().info("Соединение с UR10e закрыто.")
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = UR10eNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Остановка по Ctrl+C.")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
