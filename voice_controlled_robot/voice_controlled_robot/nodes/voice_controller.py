#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import json
import queue
import sys
import re
import os
from vosk import Model, KaldiRecognizer
import sounddevice as sd

from std_msgs.msg import String
from geometry_msgs.msg import Twist
from voice_controlled_robot.utils.audio_utils import AudioUtils

class VoiceController(Node):
    """Основной нод для голосового управления роботом."""

    def __init__(self):
        super().__init__('voice_controller')
        
        # Параметры
        self.declare_parameters(
            namespace='',
            parameters=[
                ('model_path', ''),
                ('similarity_threshold', 0.7),
                ('audio_device', 'auto'),
                ('sample_rate', 16000),
                ('publish_tool_commands', True),
                ('enable_partial_results', True)
            ]
        )
        
        # Получение параметров
        model_path_param = self.get_parameter('model_path').value
        self.similarity_threshold = self.get_parameter('similarity_threshold').value
        self.audio_device_param = self.get_parameter('audio_device').value
        self.sample_rate = self.get_parameter('sample_rate').value
        self.publish_tool_commands = self.get_parameter('publish_tool_commands').value
        self.enable_partial_results = self.get_parameter('enable_partial_results').value
        
        # Определение пути к модели
        if model_path_param:
            self.model_path = model_path_param
        else:
            # Попробуем найти модель в стандартных местах
            possible_paths = [
                os.path.expanduser('~/vosk-models/vosk-model-small-ru-0.22'),
                '/usr/share/vosk-models/vosk-model-small-ru-0.22',
                os.path.join(os.path.dirname(__file__), '../../../../share/vosk-models/vosk-model-small-ru-0.22')
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    self.model_path = path
                    break
            else:
                self.get_logger().error('❌ Модель Vosk не найдена. Установите модель:')
                self.get_logger().error('wget https://alphacephei.com/vosk/models/vosk-model-small-ru-0.22.zip')
                self.get_logger().error('unzip vosk-model-small-ru-0.22.zip')
                self.get_logger().error('И укажите путь в параметре model_path')
                raise FileNotFoundError('Модель Vosk не найдена')
        
        self.get_logger().info(f'📁 Используется модель: {self.model_path}')
        
        # Словарь инструментов
        self.tools_dict = {
            'молоток': ['молоток', 'молот', 'кувалда'],
            'отвертка': ['отвертка', 'отверточка'],
            'гаечный ключ': ['гаечный ключ', 'ключ'],
            'плоскогубцы': ['плоскогубцы', 'пассатижи'],
            'ножовка': ['ножовка', 'пила'],
            'рулетка': ['рулетка', 'метр'],
            'дрель': ['дрель', 'шуруповерт'],
            'стамеска': ['стамеска'],
            'уровень': ['уровень'],
            'нож': ['нож', 'резак']
        }
        
        # Команды управления
        self.control_commands = {
            'вперед': ['вперед', 'прямо', 'вперёд'],
            'назад': ['назад', 'обратно'],
            'влево': ['влево', 'налево'],
            'вправо': ['вправо', 'направо'],
            'стоп': ['стоп', 'остановись', 'стой'],
        }
        
        # Публикаторы
        self.tool_command_pub = self.create_publisher(String, 'voice/tool_command', 10)
        self.control_command_pub = self.create_publisher(String, 'voice/control_command', 10)
        self.velocity_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.recognized_speech_pub = self.create_publisher(String, 'voice/recognized_speech', 10)
        
        # Очередь для аудиоданных
        self.audio_queue = queue.Queue()
        
        # Инициализация аудио
        self.initialize_audio()
        
        # Таймер для обработки аудио
        self.create_timer(0.1, self.process_audio)
        
        self.get_logger().info('🎤 Голосовой контроллер запущен')

    def initialize_audio(self):
        """Инициализация аудиосистемы."""
        try:
            # Выбор аудиоустройства
            if self.audio_device_param == 'auto':
                self.audio_device = AudioUtils.pick_input_device()
            else:
                self.audio_device = int(self.audio_device_param)
            
            # Загрузка модели распознавания речи
            self.get_logger().info(f'📥 Загрузка модели Vosk из: {self.model_path}')
            self.model = Model(self.model_path)
            self.recognizer = KaldiRecognizer(self.model, self.sample_rate)
            
            # Запуск аудиопотока
            self.audio_stream = sd.RawInputStream(
                callback=self.audio_callback,
                channels=1,
                samplerate=self.sample_rate,
                device=self.audio_device,
                dtype='int16'
            )
            self.audio_stream.start()
            
            self.get_logger().info(f'✅ Аудио инициализировано (устройство: {self.audio_device})')
            
        except Exception as e:
            self.get_logger().error(f'❌ Ошибка инициализации аудио: {e}')
            raise

    def audio_callback(self, indata, frames, time, status):
        """Callback-функция для обработки аудиопотока."""
        if status:
            self.get_logger().warn(f'Аудио статус: {status}')
        self.audio_queue.put(bytes(indata))

    def find_tools_in_command(self, command):
        """Поиск инструментов в команде."""
        command = command.lower().strip()
        found_tools = []
        
        # Быстрый поиск по подстроке
        for tool, keywords in self.tools_dict.items():
            for keyword in keywords:
                if keyword in command:
                    found_tools.append(tool)
                    break
        
        if found_tools:
            return list(set(found_tools))
        
        # Поиск по схожести для случаев с ошибками
        words = re.findall(r'\w+', command)
        for word in words:
            for tool, keywords in self.tools_dict.items():
                for keyword in keywords:
                    if AudioUtils.similarity(word, keyword) >= self.similarity_threshold:
                        found_tools.append(tool)
                        break
        
        return list(set(found_tools))

    def find_control_commands(self, command):
        """Поиск команд управления в тексте."""
        command = command.lower().strip()
        
        for control_cmd, keywords in self.control_commands.items():
            for keyword in keywords:
                if keyword in command:
                    return control_cmd
        return None

    def execute_control_command(self, command):
        """Выполнение команды управления."""
        twist_msg = Twist()
        
        if command == 'вперед':
            twist_msg.linear.x = 0.2
        elif command == 'назад':
            twist_msg.linear.x = -0.2
        elif command == 'влево':
            twist_msg.angular.z = 0.5
        elif command == 'вправо':
            twist_msg.angular.z = -0.5
        elif command == 'стоп':
            twist_msg.linear.x = 0.0
            twist_msg.angular.z = 0.0
        
        self.velocity_pub.publish(twist_msg)
        self.get_logger().info(f'🚗 Выполняю команду: {command}')

    def process_audio(self):
        """Обработка аудиоданных."""
        try:
            while not self.audio_queue.empty():
                data = self.audio_queue.get()
                
                if self.recognizer.AcceptWaveform(data):
                    result = json.loads(self.recognizer.Result())
                    text = result.get("text", "").strip()
                    
                    if text:
                        self.get_logger().info(f'🗣️ Распознано: "{text}"')
                        
                        # Публикация распознанной речи
                        speech_msg = String()
                        speech_msg.data = text
                        self.recognized_speech_pub.publish(speech_msg)
                        
                        # Поиск инструментов
                        tools = self.find_tools_in_command(text)
                        if tools and self.publish_tool_commands:
                            tool_msg = String()
                            tool_msg.data = json.dumps({
                                'tools': tools,
                                'original_command': text
                            })
                            self.tool_command_pub.publish(tool_msg)
                            self.get_logger().info(f'🎯 Инструменты: {", ".join(tools)}')
                        
                        # Поиск команд управления
                        control_cmd = self.find_control_commands(text)
                        if control_cmd:
                            control_msg = String()
                            control_msg.data = control_cmd
                            self.control_command_pub.publish(control_msg)
                            self.execute_control_command(control_cmd)
                
                elif self.enable_partial_results:
                    # Вывод частичных результатов
                    partial_result = json.loads(self.recognizer.PartialResult())
                    partial_text = partial_result.get("partial", "").strip()
                    if partial_text:
                        self.get_logger().info(f'▌ Слушаю: {partial_text}', skip_first=True)
                        
        except Exception as e:
            self.get_logger().error(f'❌ Ошибка обработки аудио: {e}')

    def destroy_node(self):
        """Очистка ресурсов при завершении."""
        if hasattr(self, 'audio_stream'):
            self.audio_stream.stop()
            self.audio_stream.close()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    
    try:
        voice_controller = VoiceController()
        rclpy.spin(voice_controller)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"❌ Ошибка: {e}")
    finally:
        if 'voice_controller' in locals():
            voice_controller.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
