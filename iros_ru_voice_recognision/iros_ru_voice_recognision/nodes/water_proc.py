#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import math

import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.duration import Duration

from std_srvs.srv import Trigger
from geometry_msgs.msg import TransformStamped

from tf2_ros import Buffer, TransformListener, TransformBroadcaster


class CupsFramesNode(Node):
    """
    Нода:
      - сервис: /setup_cups_frames (std_srvs/Trigger)
      - по вызову берёт TF от base_frame до tag3_frame и tag4_frame
      - на их основе создаёт и публикует:
          * cup1_pick_frame / cup1_approach_frame
          * cup2_pick_frame / cup2_approach_frame
          * pour1_frame, pour2_frame, pour3_frame (для наливания из 1 во 2)

    ВСЕ смещения задаются в ЛОКАЛЬНЫХ координатах соответствующей кружки (AprilTag).
    """

    def __init__(self):
        super().__init__('cups_frames_node')

        # Параметры
        self.declare_parameters(
            namespace='',
            parameters=[
                ('base_frame', 'base_link'),

                # Фреймы AprilTag (ID 3 и 4)
                ('tag3_frame', 'marker3'),   # кружка 1
                ('tag4_frame', 'marker4'),   # кружка 2

                # Имена публикуемых рабочих фреймов
                ('cup1_pick_frame', 'cup1_pick'),
                ('cup1_approach_frame', 'cup1_approach'),
                ('cup2_pick_frame', 'cup2_pick'),
                ('cup2_approach_frame', 'cup2_approach'),

                ('pour1_frame', 'cup_pour_1'),
                ('pour2_frame', 'cup_pour_2'),
                ('pour3_frame', 'cup_pour_3'),

                # Локальные смещения для фреймов захвата кружек (в координатах тега)
                ('cup1_pick_offset_x', -0.025),
                ('cup1_pick_offset_y', 0.0),
                ('cup1_pick_offset_z', 0.085),

                ('cup2_pick_offset_x', -0.025),
                ('cup2_pick_offset_y', 0.0),
                ('cup2_pick_offset_z', 0.085),

                # Локальные смещения для фреймов подлёта (перед кружкой)
                ('cup1_approach_offset_x', 0.0),
                ('cup1_approach_offset_y', 0.0),
                ('cup1_approach_offset_z', 0.16),

                ('cup2_approach_offset_x', 0.0),
                ('cup2_approach_offset_y', 0.0),
                ('cup2_approach_offset_z', 0.16),

                # Локальные смещения и углы для фреймов наливания относительно кружки 2
                # pour1 — начало наклона
                ('pour1_offset_x', 0.08),
                ('pour1_offset_y', 0.12),
                ('pour1_offset_z', 0.085),
                ('pour1_roll_deg', 0.0),
                ('pour1_pitch_deg', 0.0),
                ('pour1_yaw_deg', -25.0),

                # pour2 — сильнее наклон, чуть ближе
                ('pour2_offset_x', 0.08),
                ('pour2_offset_y', 0.1),
                ('pour2_offset_z', 0.085),
                ('pour2_roll_deg', 0.0),
                ('pour2_pitch_deg', 0.0),
                ('pour2_yaw_deg', -45.0),

                # pour3 — максимальный наклон / конечная поза
                ('pour3_offset_x', 0.08),
                ('pour3_offset_y', 0.09),
                ('pour3_offset_z', 0.085),
                ('pour3_roll_deg', 0.0),
                ('pour3_pitch_deg', 0.0),
                ('pour3_yaw_deg', -75.0),
            ]
        )

        self.base_frame = self.get_parameter('base_frame').value
        self.tag3_frame = self.get_parameter('tag3_frame').value
        self.tag4_frame = self.get_parameter('tag4_frame').value

        self.cup1_pick_frame = self.get_parameter('cup1_pick_frame').value
        self.cup1_approach_frame = self.get_parameter('cup1_approach_frame').value
        self.cup2_pick_frame = self.get_parameter('cup2_pick_frame').value
        self.cup2_approach_frame = self.get_parameter('cup2_approach_frame').value

        self.pour1_frame = self.get_parameter('pour1_frame').value
        self.pour2_frame = self.get_parameter('pour2_frame').value
        self.pour3_frame = self.get_parameter('pour3_frame').value

        # offsets кружки 1
        self.cup1_pick_offset = (
            float(self.get_parameter('cup1_pick_offset_x').value),
            float(self.get_parameter('cup1_pick_offset_y').value),
            float(self.get_parameter('cup1_pick_offset_z').value),
        )
        self.cup1_approach_offset = (
            float(self.get_parameter('cup1_approach_offset_x').value),
            float(self.get_parameter('cup1_approach_offset_y').value),
            float(self.get_parameter('cup1_approach_offset_z').value),
        )

        # offsets кружки 2
        self.cup2_pick_offset = (
            float(self.get_parameter('cup2_pick_offset_x').value),
            float(self.get_parameter('cup2_pick_offset_y').value),
            float(self.get_parameter('cup2_pick_offset_z').value),
        )
        self.cup2_approach_offset = (
            float(self.get_parameter('cup2_approach_offset_x').value),
            float(self.get_parameter('cup2_approach_offset_y').value),
            float(self.get_parameter('cup2_approach_offset_z').value),
        )

        # offsets и углы для фреймов наливания (относительно кружки 2)
        self.pour_params = []
        for i in (1, 2, 3):
            off_x = float(self.get_parameter(f'pour{i}_offset_x').value)
            off_y = float(self.get_parameter(f'pour{i}_offset_y').value)
            off_z = float(self.get_parameter(f'pour{i}_offset_z').value)

            roll_deg = float(self.get_parameter(f'pour{i}_roll_deg').value)
            pitch_deg = float(self.get_parameter(f'pour{i}_pitch_deg').value)
            yaw_deg = float(self.get_parameter(f'pour{i}_yaw_deg').value)

            self.pour_params.append({
                'offset': (off_x, off_y, off_z),
                'rpy_deg': (roll_deg, pitch_deg, yaw_deg),
            })

        # TF
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.broadcaster = TransformBroadcaster(self)

        # Здесь будут храниться уже посчитанные трансформации, которые надо публиковать
        self.transforms_to_publish = []  # List[TransformStamped]

        # Сервис
        self.srv = self.create_service(
            Trigger,
            'setup_cups_frames',
            self.handle_setup_cups_frames
        )

        # Таймер для периодической публикации TF
        self.timer = self.create_timer(0.05, self.publish_transforms)

        self.get_logger().info('CupsFramesNode запущена')

    # ------------------------
    # Сервисный callback
    # ------------------------

    def handle_setup_cups_frames(self, request, response):
        """
        Вызывается по сервису /setup_cups_frames.
        Берёт TF для tag3 и tag4, создаёт рабочие фреймы и начинает их публиковать.
        """
        # Проверяем, что нужные TF доступны
        if not self._can_transform(self.base_frame, self.tag3_frame):
            msg = f'Нет TF от "{self.base_frame}" до "{self.tag3_frame}"'
            self.get_logger().warn(msg)
            response.success = False
            response.message = msg
            return response

        if not self._can_transform(self.base_frame, self.tag4_frame):
            msg = f'Нет TF от "{self.base_frame}" до "{self.tag4_frame}"'
            self.get_logger().warn(msg)
            response.success = False
            response.message = msg
            return response

        try:
            tf_tag3 = self.tf_buffer.lookup_transform(
                self.base_frame,
                self.tag3_frame,
                Time()
            )
            tf_tag4 = self.tf_buffer.lookup_transform(
                self.base_frame,
                self.tag4_frame,
                Time()
            )
        except Exception as e:
            msg = f'Ошибка lookup_transform: {e}'
            self.get_logger().error(msg)
            response.success = False
            response.message = msg
            return response

        # --- кружка 1 (источник) ---
        t_cup1_pick = TransformStamped()
        t_cup1_pick.header.frame_id = self.base_frame
        t_cup1_pick.child_frame_id = self.cup1_pick_frame

        p1 = tf_tag3.transform.translation
        q1 = tf_tag3.transform.rotation
        pick1_x, pick1_y, pick1_z = self._apply_local_offset(
            p1, q1, self.cup1_pick_offset
        )
        t_cup1_pick.transform.translation.x = pick1_x
        t_cup1_pick.transform.translation.y = pick1_y
        t_cup1_pick.transform.translation.z = pick1_z
        t_cup1_pick.transform.rotation = q1  # ориентация как у тега

        t_cup1_approach = TransformStamped()
        t_cup1_approach.header.frame_id = self.base_frame
        t_cup1_approach.child_frame_id = self.cup1_approach_frame
        appr1_x, appr1_y, appr1_z = self._apply_local_offset(
            p1, q1, self.cup1_approach_offset
        )
        t_cup1_approach.transform.translation.x = appr1_x
        t_cup1_approach.transform.translation.y = appr1_y
        t_cup1_approach.transform.translation.z = appr1_z
        t_cup1_approach.transform.rotation = q1

        # --- кружка 2 (приёмник) ---
        t_cup2_pick = TransformStamped()
        t_cup2_pick.header.frame_id = self.base_frame
        t_cup2_pick.child_frame_id = self.cup2_pick_frame

        p2 = tf_tag4.transform.translation
        q2 = tf_tag4.transform.rotation
        pick2_x, pick2_y, pick2_z = self._apply_local_offset(
            p2, q2, self.cup2_pick_offset
        )
        t_cup2_pick.transform.translation.x = pick2_x
        t_cup2_pick.transform.translation.y = pick2_y
        t_cup2_pick.transform.translation.z = pick2_z
        t_cup2_pick.transform.rotation = q2

        t_cup2_approach = TransformStamped()
        t_cup2_approach.header.frame_id = self.base_frame
        t_cup2_approach.child_frame_id = self.cup2_approach_frame
        appr2_x, appr2_y, appr2_z = self._apply_local_offset(
            p2, q2, self.cup2_approach_offset
        )
        t_cup2_approach.transform.translation.x = appr2_x
        t_cup2_approach.transform.translation.y = appr2_y
        t_cup2_approach.transform.translation.z = appr2_z
        t_cup2_approach.transform.rotation = q2

        # --- фреймы наливания (относительно кружки 2) ---
        # q2 — базовая ориентация кружки 2
        q2_tuple = (q2.x, q2.y, q2.z, q2.w)

        pour_frames = []
        pour_frame_names = [self.pour1_frame, self.pour2_frame, self.pour3_frame]

        for idx, params in enumerate(self.pour_params):
            offset = params['offset']
            roll_deg, pitch_deg, yaw_deg = params['rpy_deg']
            roll = math.radians(roll_deg)
            pitch = math.radians(pitch_deg)
            yaw = math.radians(yaw_deg)

            # позиция: смещение в локальных координатах кружки 2
            px, py, pz = self._apply_local_offset(p2, q2, offset)

            # ориентация: базовая ориентация кружки 2 + локальный поворот
            q_delta = self._euler_to_quaternion(roll, pitch, yaw)
            q_pour = self._quaternion_multiply(q2_tuple, q_delta)

            t_pour = TransformStamped()
            t_pour.header.frame_id = self.base_frame
            t_pour.child_frame_id = pour_frame_names[idx]

            t_pour.transform.translation.x = px
            t_pour.transform.translation.y = py
            t_pour.transform.translation.z = pz

            t_pour.transform.rotation.x = q_pour[0]
            t_pour.transform.rotation.y = q_pour[1]
            t_pour.transform.rotation.z = q_pour[2]
            t_pour.transform.rotation.w = q_pour[3]

            pour_frames.append(t_pour)

        # Сохраняем трансформации для будущей публикации
        self.transforms_to_publish = [
            t_cup1_pick,
            t_cup1_approach,
            t_cup2_pick,
            t_cup2_approach,
            *pour_frames,
        ]

        msg = (
            'Фреймы стаканов настроены: '
            f'{self.cup1_pick_frame}, {self.cup1_approach_frame}, '
            f'{self.cup2_pick_frame}, {self.cup2_approach_frame}, '
            f'{self.pour1_frame}, {self.pour2_frame}, {self.pour3_frame}'
        )
        self.get_logger().info(msg)

        response.success = True
        response.message = msg
        return response

    # ------------------------
    # Вспомогательные функции
    # ------------------------

    def _can_transform(self, target_frame: str, source_frame: str) -> bool:
        try:
            ok = self.tf_buffer.can_transform(
                target_frame,
                source_frame,
                Time(),
                timeout=Duration(seconds=0.5)
            )
            return ok
        except Exception as e:
            self.get_logger().warn(
                f'Ошибка can_transform({target_frame}, {source_frame}): {e}'
            )
            return False

    def _apply_local_offset(self, translation, rotation, offset):
        """
        Применяет локальное смещение offset=(dx,dy,dz), заданное в системе
        координат тега (кружки), к трансляции translation в base_frame.
        Возвращает координаты в base_frame.
        """
        dx, dy, dz = offset
        vx, vy, vz = self._rotate_vector(
            (rotation.x, rotation.y, rotation.z, rotation.w),
            (dx, dy, dz)
        )

        return (
            translation.x + vx,
            translation.y + vy,
            translation.z + vz,
        )

    @staticmethod
    def _rotate_vector(q, v):
        """
        Поворачивает вектор v=(vx,vy,vz) кватернионом q=(x,y,z,w).
        Формула без numpy: v' = v + 2*q_vec×(q_vec×v + w*v)
        """
        qx, qy, qz, qw = q
        vx, vy, vz = v

        # t = 2 * cross(q_vec, v)
        tx = 2.0 * (qy * vz - qz * vy)
        ty = 2.0 * (qz * vx - qx * vz)
        tz = 2.0 * (qx * vy - qy * vx)

        # v' = v + w * t + cross(q_vec, t)
        vpx = vx + qw * tx + (qy * tz - qz * ty)
        vpy = vy + qw * ty + (qz * tx - qx * tz)
        vpz = vz + qw * tz + (qx * ty - qy * tx)

        return vpx, vpy, vpz

    @staticmethod
    def _euler_to_quaternion(roll, pitch, yaw):
        """
        roll, pitch, yaw (рад) -> кватернион (x,y,z,w).
        """
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)

        qw = cr * cp * cy + sr * sp * sy
        qx = sr * cp * cy - cr * sp * sy
        qy = cr * sp * cy + sr * cp * sy
        qz = cr * cp * sy - sr * sp * cy

        return (qx, qy, qz, qw)

    @staticmethod
    def _quaternion_multiply(q1, q2):
        """
        Перемножение кватернионов q = q1 * q2.
        """
        x1, y1, z1, w1 = q1
        x2, y2, z2, w2 = q2

        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2

        return (x, y, z, w)

    # ------------------------
    # Публикация TF
    # ------------------------

    def publish_transforms(self):
        """
        Периодически публикует заранее посчитанные TransformStamped.
        Если сервис ещё не вызывался или он завершился неуспешно — список пуст.
        """
        if not self.transforms_to_publish:
            return

        now = self.get_clock().now().to_msg()
        for t in self.transforms_to_publish:
            t.header.stamp = now

        self.broadcaster.sendTransform(self.transforms_to_publish)


def main(args=None):
    rclpy.init(args=args)
    node = None

    try:
        node = CupsFramesNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f'Ошибка в CupsFramesNode: {e}', file=sys.stderr)
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
