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

from ur_assist.srv import GripperAction, GoToFrame


class VoiceCommandExecutor(Node):
    """
    Нода: слушает voice/command и в зависимости от команды
    запускает сценарий движения манипулятора через сервисы:
      - /gripper_action (ur_assist/srv/GripperAction)
      - /go_to_frame   (ur_assist/srv/GoToFrame)

    Архитектура:
      - очередь сценариев (command -> list шагов)
      - один активный сценарий
      - выполнение шагов через асинхронные сервисные вызовы
        и периодический таймер.
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

                ('hammer_pick_frame', 'hoba_target'),
                ('hammer_place_frame', 'pose_up'),

                ('water_pick_frame', 'marker_water_pick'),
                ('water_place_frame', 'marker_water_place'),

                ('up_frame', 'pose_up'),
                ('forward_frame', 'pose_forward'),

                # Если False — игнорировать новые команды, пока идёт сценарий
                ('queue_commands', False),
            ]
        )

        self.base_frame = self.get_parameter('base_frame').value
        self.home_frame = self.get_parameter('home_frame').value
        self.podat_frame = self.get_parameter('podat_frame').value


        self.hammer_pick_frame = self.get_parameter('hammer_pick_frame').value
        self.hammer_place_frame = self.get_parameter('hammer_place_frame').value

        self.water_pick_frame = self.get_parameter('water_pick_frame').value
        self.water_place_frame = self.get_parameter('water_place_frame').value

        self.up_frame = self.get_parameter('up_frame').value
        self.forward_frame = self.get_parameter('forward_frame').value

        self.queue_commands = bool(self.get_parameter('queue_commands').value)

        # TF
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Сценарии: список шагов (тип, аргументы)
        # тип: "go" или "gripper"
        self.scenarios = self._build_scenarios()

        # Клиенты к сервисам
        self.gripper_client = self.create_client(GripperAction, '/gripper_action')
        self.goto_client = self.create_client(GoToFrame, '/go_to_frame')

        # Ждём сервисы на старте (блокирующе, но до spin это нормально)
        self.get_logger().info('Ожидание сервиса /gripper_action ...')
        self.gripper_client.wait_for_service()
        self.get_logger().info('Ожидание сервиса /go_to_frame ...')
        self.goto_client.wait_for_service()
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

        # Таймер "тикера" сценариев
        self.tick_timer = self.create_timer(0.05, self._tick)

        self.get_logger().info('Нода VoiceCommandExecutor запущена')

    def _build_scenarios(self):
        """
        Собирает словарь сценариев на основе параметров фреймов.
        Можно править под свои задачи.
        """
        scenarios = {}

        scenarios['молоток'] = [
            ('go', self.up_frame),
            ('gripper', True),   # открыть
            ('go', self.hammer_pick_frame),
            ('gripper', False),  # закрыть
            ('go', self.hammer_place_frame),
            ('go', self.podat_frame),
        ]

        # # Сценарий "воды"
        # scenarios['воды'] = [
        #     ('go', self.water_pick_frame),
        #     ('gripper', False),
        #     ('go', self.water_place_frame),
        #     ('gripper', True),
        #     ('go', self.home_frame),
        # ]

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
            # Time() с нулевым временем — "latest"
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
        else:
            self.get_logger().error(f'Неизвестный тип шага: {step_type}')
            self._abort_current_scenario(reason=f'unknown_step_type:{step_type}')

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

    def _finish_current_scenario(self):
        self.get_logger().info(f'Сценарий "{self.current_command}" завершён')
        self._publish_status(f'done:{self.current_command}')

        self.current_command = None
        self.current_steps = None
        self.current_step_index = 0
        self.current_future = None
        self.cancel_requested = False

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

    # ------------------------
    # Таймер-тикер
    # ------------------------

    def _tick(self):
        """
        Периодический "тик":
          - если есть активный сервисный вызов, ждём его завершения;
          - после завершения шага запускаем следующий.
        """
        if self.current_command is None:
            return

        # Если нет активного future — ничего не делаем
        if self.current_future is None:
            return

        # Если future ещё не готов — ждём
        if not self.current_future.done():
            return

        # Обрабатываем результат вызова
        try:
            _ = self.current_future.result()
        except Exception as exc:
            self.get_logger().error(
                f'Ошибка при выполнении шага сценария "{self.current_command}": {exc}'
            )
            self._abort_current_scenario(reason='service_error')
            return

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
