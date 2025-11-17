#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import json
import time
import os
import math
import queue
import re
import sys
from difflib import SequenceMatcher

import sounddevice as sd
import vosk
#import URBasic
#import URBasic.robotModel
#import URBasic.urScriptExt
from ament_index_python.packages import get_package_share_directory

class VoiceController(Node):
    """Основной узел ROS2 для голосового управления роботом."""
    
    # Словарь инструментов
    TOOLS_DICT = {
        'посмотри': ['посмотри', 'изучи', 'осмотри', 'взглянги'],
        'собачка': ['собака', 'собаку', 'собачку'],
        'назад': ['вернись', 'начало', 'назад']
    }
    
    def __init__(self):
        super().__init__('voice_controller')
        
        # Получаем путь к ресурсам пакета
        package_share_path = get_package_share_directory('voice_controlled_robot')
        self.resources_path = os.path.join(package_share_path, 'resources')
        
        # Добавляем URBasic в путь Python
        urbasic_path = os.path.join(self.resources_path, 'URBasic')
        if urbasic_path not in sys.path:
            sys.path.insert(0, urbasic_path)
        
        print(f"🔧 Ресурсы пакета: {self.resources_path}")
        print(f"🔧 Путь URBasic: {urbasic_path}")
        
        # Инициализация переменных
        self.robot = None
        self.robot_model = None
        self.audio_queue = queue.Queue()
        self.sample_rate = None
        self.audio_device = None
        self.recognizer = None
        
        # Загрузка параметров
        self.declare_parameters(
            namespace='',
            parameters=[
                ('robot_host', '192.168.0.100'),
                ('waiting_pose', [-0.19, -0.315, 0.350, 2.783, -1.353, 0.03]),
                ('dog_pose', [-0.828, -0.624, 0.490, 2.003, -1.873, 0.08]),
                ('looking_poses', [
                    [-0.22, -0.873, 1, 1.539, -0.749, 0.584],
                    [0.185, -0.893, 1.1, 1.614, -0.831, 0.133],
                    [-0.610, -0.650, 0.890, 1.607, -0.822, 0.750],
                    [-0.228, -0.388, 1.26, 1.435, -0.730, 0.592],
                    [-0.22, -0.873, 1, 1.539, -0.749, 0.584]
                ]),
                ('sample_rate', 16000),
                ('similarity_threshold', 0.7),
                ('position_timeout', 15.0),
                ('pause_duration', 2.0)
            ]
        )
        
        # Получение параметров
        self.robot_host = self.get_parameter('robot_host').value
        self.waiting_pose = self.get_parameter('waiting_pose').value
        self.dog_pose = self.get_parameter('dog_pose').value
        self.looking_poses = self.get_parameter('looking_poses').value
        self.sample_rate = self.get_parameter('sample_rate').value
        self.similarity_threshold = self.get_parameter('similarity_threshold').value
        self.position_timeout = self.get_parameter('position_timeout').value
        self.pause_duration = self.get_parameter('pause_duration').value
        
        self.get_logger().info("Инициализация Voice Controller...")

    @staticmethod
    def normalize_angle(angle):
        """Нормализует угол в диапазон [-pi, pi]."""
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle

    def check_position(self, current_pos, target_pos, linear_tolerance=0.001, angle_tolerance=0.01):
        """Проверяет, достигнута ли целевая позиция."""
        position_reached = True
        
        self.get_logger().debug("Проверка позиции:")
        for i, (current, target) in enumerate(zip(current_pos, target_pos)):
            if i < 3:  # Позиция (x, y, z)
                diff = abs(current - target)
                if diff > linear_tolerance:
                    position_reached = False
                status = '✓' if diff <= linear_tolerance else '✗'
                self.get_logger().debug(
                    f"Ось {i}(лин.): текущая={current:08.6f}, "
                    f"целевая={target:08.6f}, разница={diff:08.6f} {status}"
                )
            else:  # Ориентация (rx, ry, rz)
                norm_current = self.normalize_angle(current)
                norm_target = self.normalize_angle(target)
                diff = abs(-norm_current - norm_target)
                if diff > angle_tolerance:
                    position_reached = False
                status = '✓' if diff <= angle_tolerance else '✗'
                self.get_logger().debug(
                    f"Ось {i}(угл.): текущая={current:08.6f}→{norm_current:08.6f}, "
                    f"целевая={target:08.6f}→{norm_target:08.6f}, "
                    f"разница={diff:08.6f} {status}"
                )
        
        self.get_logger().debug(f"Позиция достигнута: {position_reached}")
        return position_reached

    def wait_for_position(self, target_pos, position_name="", timeout=15.0, check_interval=0.05):
        """Ожидает достижения целевой позиции с таймаутом."""
        start_time = time.time()
        attempt = 0
        
        while time.time() - start_time < timeout:
            attempt += 1
            current_pos = self.robot.get_actual_tcp_pose_custom()
            
            if self.check_position(current_pos, target_pos):
                self.get_logger().info(f"✓ Позиция '{position_name}' успешно достигнута!")
                return True
            
            if attempt % 10 == 0:
                elapsed = time.time() - start_time
                self.get_logger().info(f"Прогресс: {elapsed:.1f}с из {timeout}с")
            
            time.sleep(check_interval)
        
        self.get_logger().error(f"✗ Таймаут: позиция '{position_name}' не достигнута за {timeout} секунд")
        return False

    def initialize_robot(self):
        """Инициализация соединения с роботом."""
        self.get_logger().info("Инициализация UR робота")
        self.robot_model = URBasic.robotModel.RobotModel()
        self.robot = URBasic.urScriptExt.UrScriptExt(
            host=self.robot_host, 
            robotModel=self.robot_model
        )
        self.robot.init_realtime_control()
        
        current_pos = self.robot.get_actual_tcp_pose_custom()
        self.get_logger().info(
            f'Текущая позиция робота: '
            f'[{current_pos[0]:08.6f}, {current_pos[1]:08.6f}, {current_pos[2]:08.6f}, '
            f'{current_pos[3]:08.6f}, {current_pos[4]:08.6f}, {current_pos[5]:08.6f}]'
        )
        
        return self.robot

    def move_to_position(self, target_pos, position_name):
        """Перемещение робота в указанную позицию."""
        self.get_logger().info(f"Переход в позицию '{position_name}': {target_pos}")
        self.robot.set_realtime_pose(target_pos)
        
        if not self.wait_for_position(target_pos, position_name, self.position_timeout):
            self.get_logger().error(f"Ошибка: не удалось достичь позиции '{position_name}'")
            return False
        return True

    @staticmethod
    def pause_at_position(position_name, duration=2.0):
        """Пауза в достигнутой позиции."""
        print(f"⏸️  Пауза {duration} секунд в позиции '{position_name}'...")
        time.sleep(duration)
        print(f"▶️  Пауза завершена, продолжаем работу.")

    @staticmethod
    def similarity(a, b):
        """Вычисляет схожесть двух строк."""
        return SequenceMatcher(None, a, b).ratio()

    def find_tools_in_command(self, command, similarity_threshold=0.7):
        """Поиск инструментов в голосовой команде."""
        command = command.lower().strip()
        found_tools = []
        
        # Быстрый поиск по подстроке
        for tool, keywords in self.TOOLS_DICT.items():
            for keyword in keywords:
                if keyword in command:
                    found_tools.append(tool)
                    break
        
        if found_tools:
            return list(set(found_tools))
        
        # Поиск по схожести для случаев с ошибками
        words = re.findall(r'\w+', command)
        for word in words:
            for tool, keywords in self.TOOLS_DICT.items():
                for keyword in keywords:
                    if self.similarity(word, keyword) >= similarity_threshold:
                        found_tools.append(tool)
                        break
        
        return list(set(found_tools))

    def pick_input_device(self):
        """Выбор аудио устройства для записи."""
        devices = sd.query_devices()
        
        # Приоритет: pulse → default → любой с входом
        for i, device in enumerate(devices):
            if (device["max_input_channels"] > 0 and 
                device["name"].lower() == "pulse"):
                self.get_logger().info(f"Выбрано аудио устройство: {device['name']}")
                return i
                
        for i, device in enumerate(devices):
            if (device["max_input_channels"] > 0 and 
                device["name"].lower() == "default"):
                self.get_logger().info(f"Выбрано аудио устройство: {device['name']}")
                return i
                
        for i, device in enumerate(devices):
            if (device["max_input_channels"] > 0 and 
                "hdmi" not in device["name"].lower()):
                self.get_logger().info(f"Выбрано аудио устройство: {device['name']}")
                return i
                
        raise RuntimeError("Нет входных аудиоустройств")

    def audio_callback(self, indata, frames, time_info, status):
        """Callback функция для обработки аудио потока."""
        if status:
            self.get_logger().warn(f"Аудио статус: {status}")
        self.audio_queue.put(bytes(indata))

    def initialize_audio(self, model_path):
        """Инициализация аудио системы."""
        self.audio_device = self.pick_input_device()
        
        model = vosk.Model(model_path)
        self.recognizer = vosk.KaldiRecognizer(model, self.sample_rate)
        
        return sd.RawInputStream(
            callback=self.audio_callback,
            channels=1,
            samplerate=self.sample_rate,
            device=self.audio_device,
            dtype='int16'
        )

    def execute_command(self, tools):
        """Выполнение команды на основе распознанных инструментов."""
        if not tools:
            return
            
        action = tools[0]
        
        if action == 'собачка':
            self.move_to_position(self.dog_pose, "Позиция собачки")
            self.pause_at_position("Позиция собачки", self.pause_duration)
            
        elif action == 'назад':
            self.move_to_position(self.waiting_pose, "Начальная позиция")
            
        elif action == 'посмотри':
            for i, pose in enumerate(self.looking_poses):
                self.move_to_position(pose, f"Позиция осмотра {i+1}")
                self.pause_at_position(f"Позиция осмотра {i+1}", 1.0)

    def run(self):
        """Основной цикл работы узла."""
        try:
            # Инициализация робота
            self.initialize_robot()
            
            # Переход в начальную позицию
            self.get_logger().info("Переход в начальную позицию...")
            if not self.move_to_position(self.waiting_pose, "Начальная позиция"):
                return
            
            # Инициализация аудио
            model_path = os.path.join(
                get_package_share_directory('voice_controlled_robot'),
                'resources',
                'vosk-model-small-ru-0.22'
            )
            
            # Если модель не найдена в resources, используем локальную
            if not os.path.exists(model_path):
                model_path = "model_stt/vosk-model-small-ru-0.22"
                self.get_logger().warn(f"Модель не найдена в пакете, используем локальную: {model_path}")
            
            with self.initialize_audio(model_path) as stream:
                self.get_logger().info("🎤 Голосовой ассистент запущен. Говорите...")
                
                while rclpy.ok():
                    data = self.audio_queue.get()
                    if self.recognizer.AcceptWaveform(data):
                        result = json.loads(self.recognizer.Result())
                        text = result.get("text", "").strip()
                        
                        if text:
                            self.get_logger().info(f"🗣️ Команда: {text}")
                            
                            tools = self.find_tools_in_command(text, self.similarity_threshold)
                            
                            if tools:
                                self.get_logger().info(f"🎯 Распознаны действия: {', '.join(tools)}")
                                self.execute_command(tools)
                            else:
                                self.get_logger().warn("❌ Инструменты не распознаны")
                                
                    else:
                        partial_result = json.loads(self.recognizer.PartialResult())
                        partial_text = partial_result.get("partial", "").strip()
                        if partial_text:
                            print(f"▌ Слушаю: {partial_text}", end='\r')
                            
        except KeyboardInterrupt:
            self.get_logger().info("✅ Ассистент остановлен")
        except Exception as e:
            self.get_logger().error(f"❌ Ошибка: {e}")
        finally:
            if self.robot:
                self.robot.close()

def main(args=None):
    rclpy.init(args=args)
    
    voice_controller = VoiceController()
    
    try:
        voice_controller.run()
    except Exception as e:
        voice_controller.get_logger().error(f"Ошибка в main: {e}")
    finally:
        voice_controller.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
