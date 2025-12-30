#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
from collections import deque

import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.duration import Duration
from std_msgs.msg import String

from tf2_ros import Buffer, TransformListener

from std_srvs.srv import Trigger

from iros_assistant_bringup.srv import GripperAction, GoToFrame, DetectObject


class VoiceCommandExecutor(Node):
    """
    Нода: слушает voice/command и в зависимости от команды
    запускает сценарий движения манипулятора через сервисы:
      - /gripper_action (iros_assistant_bringup/srv/GripperAction)
      - /go_to_frame   (iros_assistant_bringup/srv/GoToFrame)
      - /detect_object (iros_assistant_bringup/srv/DetectObject)
      - /setup_cups_frames (std_srvs/srv/Trigger)

    Шаги сценария:
      - ('go', frame_name)
      - ('gripper', True/False)
      - ('detect', class_name, duration)
      - ('wait', seconds)
      - ('setup_cups',)
    """

    def __init__(self):
        super().__init__('voice_command_executor')

        # Параметры фреймов и поведения
        self.declare_parameters(
            namespace='',
            parameters=[
                ('base_frame', 'base_link'),
                ('home_frame', 'home'),
                ('podat_frame', 'pose_podat'),
                ('watering_frame', 'pose_watering'),

                ('hammer_pick_frame', 'hoba_target'),
                ('hammer_place_frame', 'pose_up'),

                ('water_pick_frame', 'marker_water_pick'),
                ('water_place_frame', 'marker_water_place'),

                ('up_frame', 'pose_up'),
                ('forward_frame', 'pose_forward'),

                # Для сценария с детекцией отвертки
                ('screwdriver_frame', 'hoba_target'),

                # Фреймы для сценария с двумя кружками (cups_frames_node)
                ('cup1_pick_frame', 'cup1_pick'),
                ('cup1_approach_frame', 'cup1_approach'),
                ('cup2_pick_frame', 'cup2_pick'),
                ('cup2_approach_frame', 'cup2_approach'),
                ('pour1_frame', 'cup_pour_1'),
                ('pour2_frame', 'cup_pour_2'),
                ('pour3_frame', 'cup_pour_3'),

                # Сколько секунд трекать объект в detect_object
                ('detect_duration', 10.0),

                # Если False — игнорировать новые команды, пока идёт сценарий
                ('queue_commands', False),
            ]
        )

        self.base_frame = self.get_parameter('base_frame').value
        self.home_frame = self.get_parameter('home_frame').value
        self.podat_frame = self.get_parameter('podat_frame').value
        self.watering_frame = self.get_parameter('watering_frame').value

        self.hammer_pick_frame = self.get_parameter('hammer_pick_frame').value
        self.hammer_place_frame = self.get_parameter('hammer_place_frame').value

        self.water_pick_frame = self.get_parameter('water_pick_frame').value
        self.water_place_frame = self.get_parameter('water_place_frame').value

        self.up_frame = self.get_parameter('up_frame').value
        self.forward_frame = self.get_parameter('forward_frame').value

        self.screwdriver_frame = self.get_parameter('screwdriver_frame').value

        self.cup1_pick_frame = self.get_parameter('cup1_pick_frame').value
        self.cup1_approach_frame = self.get_parameter('cup1_approach_frame').value
        self.cup2_pick_frame = self.get_parameter('cup2_pick_frame').value
        self.cup2_approach_frame = self.get_parameter('cup2_approach_frame').value
        self.pour1_frame = self.get_parameter('pour1_frame').value
        self.pour2_frame = self.get_parameter('pour2_frame').value
        self.pour3_frame = self.get_parameter('pour3_frame').value

        self.detect_duration = float(self.get_parameter('detect_duration').value)

        self.queue_commands = bool(self.get_parameter('queue_commands').value)

        # TF
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Сценарии: список шагов (тип, аргументы)
        # тип шага: "go", "gripper", "detect", "wait", "setup_cups"
        self.scenarios = self._build_scenarios()

        # Клиенты к сервисам
        self.gripper_client = self.create_client(GripperAction, '/gripper_action')
        self.goto_client = self.create_client(GoToFrame, '/go_to_frame')
        self.detect_client = self.create_client(DetectObject, '/detect_object')
        self.cups_setup_client = self.create_client(Trigger, '/setup_cups_frames')

        # Ждём сервисы на старте (блокирующе, но до spin это нормально)
        self.get_logger().info('Ожидание сервиса /gripper_action ...')
        self.gripper_client.wait_for_service()
        self.get_logger().info('Ожидание сервиса /go_to_frame ...')
        self.goto_client.wait_for_service()
        self.get_logger().info('Ожидание сервиса /detect_object ...')
        self.detect_client.wait_for_service()
        self.get_logger().info('Ожидание сервиса /setup_cups_frames ...')
        self.cups_setup_client.wait_for_service()
        self.get_logger().info('Сервисы доступны')

        # Подписчик на распознанные команды
        self.command_sub = self.create_subscription(
            String,
            'voice/command',
            self.command_callback,
            10
        )

        # Публикатор статуса/отладки
        self.status_pub = self.create_publisher(String, 'voice/executor_status', 10)

        # Очередь и состояние текущего сценария
        self.scenario_queue = deque()     # элементы: (command_name: str, steps: list)
        self.current_command = None       # имя команды (каноническое)
        self.current_steps = None         # list шагов
        self.current_step_index = 0       # индекс шага
        self.current_future = None        # Future сервисного вызова
        self.cancel_requested = False     # флаг остановки

        # Для шага "wait"
        self.wait_until = None            # rclpy.time.Time, когда ожидание заканчивается

        # Таймер "тикера" сценариев
        self.tick_timer = self.create_timer(0.05, self._tick)

        self.get_logger().info('Нода VoiceCommandExecutor запущена')

    def _build_scenarios(self):
        """
        Собирает словарь сценариев на основе параметров фреймов.
        Можно править под свои задачи.
        """
        scenarios = {}

        # Молоток
        scenarios['молоток'] = [
            ('go', self.up_frame),
            ('gripper', True),                             # открыть
            ('detect', 'hammer', self.detect_duration),    # вызвать /detect_object
            ('wait', 3.0),                                 # подождать 3 секунды
            ('go', self.hammer_pick_frame),                # сюда должен публиковаться TF от детектора
            ('gripper', False),                            # закрыть
            ('go', self.up_frame),
            ('go', self.podat_frame),
        ]

        # Отвёртка
        scenarios['отвёртка'] = [
            ('go', self.up_frame),
            ('gripper', True),
            ('detect', 'screwdriver', self.detect_duration),
            ('wait', 3.0),
            ('go', self.screwdriver_frame),
            ('gripper', False),
            ('go', self.up_frame),
            ('go', self.podat_frame),
        ]

        # Новое действие: "налей"
        scenarios['воды'] = [
            # 1) Настроить рабочие фреймы кружек по AprilTag (через cups_frames_node)
            ('go', self.watering_frame),
            ('setup_cups',),
            ('wait', 0.5),  # небольшая пауза, чтобы TF успели появиться

            # 2) Взять первую кружку
            ('gripper', True),                   # открыть хват
            ('go', self.cup1_approach_frame),    # подлёт к кружке 1
            ('go', self.cup1_pick_frame),        # захват
            ('gripper', False),                  # закрыть, взять кружку

            # 3) Наливание во вторую кружку через три позы
            ('go', self.pour1_frame),
            ('wait', 1.0),
            ('go', self.pour2_frame),
            ('wait', 1.0),
            ('go', self.pour3_frame),
            ('wait', 1.0),
            # возвращаемся из максимального наклона обратно
            ('go', self.pour2_frame),
            ('go', self.pour1_frame),

            # 4) Положить первую кружку обратно
            ('go', self.cup1_pick_frame),
            ('gripper', True),                   # отпустить кружку
            ('go', self.cup1_approach_frame),    # подлёт к кружке 1

            # 5) Взять вторую кружку и поднести
            ('go', self.cup2_approach_frame),
            ('go', self.cup2_pick_frame),
            ('gripper', False),                  # взять кружку 2
            ('go', self.podat_frame),            # поднести
        ]

        # Простейшие команды
        scenarios['поднеси'] = [
            ('go', self.podat_frame),
        ]

        scenarios['вперед'] = [
            ('go', self.forward_frame),
        ]

        scenarios['открой'] = [
            ('gripper', True),
        ]

        scenarios['закрой'] = [
            ('gripper', False),
        ]

        scenarios['вверх'] = [
            ('go', self.up_frame),
        ]

        return scenarios

    # ------------------------
    # Проверка наличия TF
    # ------------------------

    def _check_frame_exists(self, frame: str) -> bool:
        """
        Проверяет, существует ли TF от base_frame до frame.
        Если трансформа нет — пишет warning и возвращает False.
        """
        try:
            ok = self.tf_buffer.can_transform(
                self.base_frame,
                frame,
                Time(),
                timeout=Duration(seconds=0.0)
            )
        except Exception as e:
            self.get_logger().warn(
                f'Ошибка при проверке TF для "{frame}" относительно "{self.base_frame}": {e}'
            )
            return False

        if not ok:
            self.get_logger().warn(
                f'TF для фрейма "{frame}" относительно "{self.base_frame}" не найден. '
                f'Сценарий "{self.current_command}" будет прерван.'
            )
            return False

        return True

    # ------------------------
    # Подписчик на voice/command
    # ------------------------

    def command_callback(self, msg: String):
        cmd = msg.data.strip().lower()
        if not cmd:
            return

        # Специальная команда "стоп"
        if cmd == 'стоп':
            self._handle_stop_command()
            return

        # Если команды нет в сценариях — игнорируем
        if cmd not in self.scenarios:
            self.get_logger().warn(f'Нет сценария для команды "{cmd}", игнорирую')
            self._publish_status(f'unknown_command:{cmd}')
            return

        # Уже идёт сценарий
        if self.current_command is not None:
            if self.queue_commands:
                self.scenario_queue.append((cmd, list(self.scenarios[cmd])))
                self.get_logger().info(f'Команда "{cmd}" поставлена в очередь')
                self._publish_status(f'queued:{cmd}')
            else:
                self.get_logger().info(
                    f'Команда "{cmd}" проигнорирована: сценарий "{self.current_command}" ещё выполняется'
                )
            return

        # Если свободны — запускаем сразу
        self._start_new_scenario(cmd, list(self.scenarios[cmd]))

    def _handle_stop_command(self):
        """
        Обработка голосовой команды "стоп":
          - ставим флаг отмены текущего сценария;
          - очищаем очередь.
        """
        self.cancel_requested = True
        self.scenario_queue.clear()

        if self.current_command is None:
            self.get_logger().info('Команда "стоп": активного сценария нет, просто очищена очередь')
        else:
            self.get_logger().info(f'Команда "стоп": запрошена остановка сценария "{self.current_command}"')

        self._publish_status('stop_requested')

    # ------------------------
    # Управление сценариями
    # ------------------------

    def _start_new_scenario(self, command_name: str, steps: list):
        self.current_command = command_name
        self.current_steps = steps
        self.current_step_index = 0
        self.current_future = None
        self.cancel_requested = False
        self.wait_until = None

        self.get_logger().info(f'Запуск сценария для команды "{command_name}"')
        self._publish_status(f'start:{command_name}')

        # Запускаем первый шаг
        self._start_next_step()

    def _start_next_step(self):
        """
        Запуск следующего шага сценария (если есть).
        Вызывается при старте сценария и после завершения каждого сервисного вызова.
        """
        if self.current_command is None or self.current_steps is None:
            return

        # Проверка на остановку
        if self.cancel_requested:
            self._abort_current_scenario(reason='cancel_requested')
            return

        if self.current_step_index >= len(self.current_steps):
            # Сценарий закончен
            self._finish_current_scenario()
            return

        step = self.current_steps[self.current_step_index]
        step_type = step[0]

        if step_type == 'go':
            frame = step[1]

            # Проверяем, что TF до этого фрейма существует
            if not self._check_frame_exists(frame):
                self._abort_current_scenario(reason=f'no_tf:{frame}')
                return

            self._call_go_to_frame(frame)

        elif step_type == 'gripper':
            open_flag = bool(step[1])
            self._call_gripper(open_flag)

        elif step_type == 'detect':
            # step: ('detect', class_name, duration)
            class_name = step[1]
            duration = step[2] if len(step) > 2 else self.detect_duration
            self._call_detect_object(class_name, duration)

        elif step_type == 'wait':
            duration = float(step[1]) if len(step) > 1 else 0.0
            self._start_wait(duration)

        elif step_type == 'setup_cups':
            self._call_setup_cups()

        else:
            self.get_logger().error(f'Неизвестный тип шага: {step_type}')
            self._abort_current_scenario(reason=f'unknown_step_type:{step_type}')

    def _start_wait(self, duration: float):
        """
        Инициализация шага ожидания.
        """
        if duration <= 0.0:
            self.get_logger().info(
                f'Шаг {self.current_step_index}: wait({duration} s) — пропускаю (<=0)'
            )
            # Сразу переходим к следующему шагу
            self.current_step_index += 1
            self._start_next_step()
            return

        self.wait_until = self.get_clock().now() + Duration(seconds=duration)
        self.get_logger().info(
            f'Шаг {self.current_step_index}: wait({duration} s) — ожидание начато'
        )

    def _call_go_to_frame(self, frame: str):
        req = GoToFrame.Request()
        req.frame = frame
        self.get_logger().info(f'Шаг {self.current_step_index}: go_to_frame("{frame}")')
        self.current_future = self.goto_client.call_async(req)

    def _call_gripper(self, open_flag: bool):
        req = GripperAction.Request()
        req.open = open_flag
        action = 'open' if open_flag else 'close'
        self.get_logger().info(f'Шаг {self.current_step_index}: gripper({action})')
        self.current_future = self.gripper_client.call_async(req)

    def _call_detect_object(self, class_name: str, duration: float):
        req = DetectObject.Request()
        req.class_name = class_name
        req.duration = float(duration)
        self.get_logger().info(
            f'Шаг {self.current_step_index}: detect_object(class_name="{class_name}", duration={duration})'
        )
        self.current_future = self.detect_client.call_async(req)

    def _call_setup_cups(self):
        req = Trigger.Request()
        self.get_logger().info(
            f'Шаг {self.current_step_index}: setup_cups_frames()'
        )
        self.current_future = self.cups_setup_client.call_async(req)

    def _finish_current_scenario(self):
        self.get_logger().info(f'Сценарий "{self.current_command}" завершён')
        self._publish_status(f'done:{self.current_command}')

        self.current_command = None
        self.current_steps = None
        self.current_step_index = 0
        self.current_future = None
        self.cancel_requested = False
        self.wait_until = None

        # Если есть сценарии в очереди — запускаем следующий
        if self.scenario_queue:
            next_cmd, next_steps = self.scenario_queue.popleft()
            self._start_new_scenario(next_cmd, next_steps)

    def _abort_current_scenario(self, reason: str):
        self.get_logger().warn(f'Сценарий "{self.current_command}" прерван, причина: {reason}')
        self._publish_status(f'abort:{self.current_command}:{reason}')

        self.current_command = None
        self.current_steps = None
        self.current_step_index = 0
        self.current_future = None
        self.cancel_requested = False
        self.wait_until = None

    # ------------------------
    # Таймер-тикер
    # ------------------------

    def _tick(self):
        """
        Периодический "тик":
          - если есть активный сервисный вызов, ждём его завершения;
          - если текущий шаг — 'wait', проверяем истечение времени;
          - после завершения шага запускаем следующий.
        """
        if self.current_command is None or self.current_steps is None:
            return

        # Если запрошен стоп — прерываем
        if self.cancel_requested:
            self._abort_current_scenario(reason='cancel_requested')
            return

        if self.current_step_index >= len(self.current_steps):
            return

        step = self.current_steps[self.current_step_index]
        step_type = step[0]

        # Обработка шага ожидания
        if step_type == 'wait':
            if self.wait_until is None:
                # На всякий случай: если не инициализировано, считаем, что ждать не надо
                self.get_logger().warn(
                    f'wait_without_wait_until на шаге {self.current_step_index}, пропускаю шаг'
                )
                self.current_step_index += 1
                self._start_next_step()
                return

            now = self.get_clock().now()
            if now >= self.wait_until:
                self.get_logger().info(
                    f'Шаг {self.current_step_index}: wait завершён'
                )
                self.wait_until = None
                self.current_step_index += 1
                self._start_next_step()
            return

        # Для остальных шагов ждём завершения текущего future
        if self.current_future is None:
            return

        if not self.current_future.done():
            return

        # Обрабатываем результат вызова
        try:
            result = self.current_future.result()
        except Exception as exc:
            self.get_logger().error(
                f'Ошибка при выполнении шага сценария "{self.current_command}": {exc}'
            )
            self._abort_current_scenario(reason='service_error')
            return

        # Специальная обработка для detect_object: проверяем accepted
        if step_type == 'detect':
            accepted = getattr(result, 'accepted', True)
            message = getattr(result, 'message', '')
            if not accepted:
                self.get_logger().warn(
                    f'detect_object не принял запрос для "{step[1]}": {message}'
                )
                self._abort_current_scenario(reason='detect_rejected')
                return
            else:
                self.get_logger().info(
                    f'detect_object ответил accepted=True для "{step[1]}": {message}'
                )

        # Специальная обработка для setup_cups_frames: проверяем success
        if step_type == 'setup_cups':
            success = getattr(result, 'success', True)
            message = getattr(result, 'message', '')
            if not success:
                self.get_logger().warn(
                    f'setup_cups_frames неуспешен: {message}'
                )
                self._abort_current_scenario(reason='setup_cups_failed')
                return
            else:
                self.get_logger().info(
                    f'setup_cups_frames успешно: {message}'
                )

        # Шаг успешно выполнен, переходим к следующему
        self.current_future = None
        self.current_step_index += 1
        self._start_next_step()

    # ------------------------
    # Вспомогательное
    # ------------------------

    def _publish_status(self, text: str):
        msg = String()
        msg.data = text
        self.status_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    executor_node = None

    try:
        executor_node = VoiceCommandExecutor()
        rclpy.spin(executor_node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f'Ошибка в VoiceCommandExecutor: {e}', file=sys.stderr)
    finally:
        if executor_node is not None:
            executor_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
