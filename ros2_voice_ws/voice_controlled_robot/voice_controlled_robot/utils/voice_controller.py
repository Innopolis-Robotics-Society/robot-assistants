#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import json
import time
import os
from ament_index_python.packages import get_package_share_directory

from voice_controlled_robot.utils.audio_utils import AudioUtils
from voice_controlled_robot.utils.robot_utils import RobotUtils
from voice_controlled_robot.utils.speech_utils import SpeechUtils

class VoiceController(Node):
    """Основной узел ROS2 для голосового управления роботом."""
    
    def __init__(self):
        super().__init__('voice_controller')
        
        # Загрузка параметров
        self.declare_parameters(
            namespace='',
            parameters=[
                ('robot_host', '192.168.0.100'),
                ('waiting_pose', [-0.19, -0.315, 0.350, 2.783, -1.353, 0.03]),
                ('dog_pose', [-0.828, -0.624, 0.490, 2.003, -1.873, 0.08]),
                ('looking_poses', []),
                ('sample_rate', 16000),
                ('similarity_threshold', 0.7),
                ('position_timeout', 15.0),
                ('pause_duration', 2.0)
            ]
        )
        
        # Инициализация утилит
        self.audio_utils = AudioUtils(self)
        self.robot_utils = RobotUtils(self)
        self.speech_utils = SpeechUtils(self)
        
        # Получение параметров
        self.robot_host = self.get_parameter('robot_host').value
        self.waiting_pose = self.get_parameter('waiting_pose').value
        self.dog_pose = self.get_parameter('dog_pose').value
        self.looking_poses = self.get_parameter('looking_poses').value
        self.sample_rate = self.get_parameter('sample_rate').value
        self.similarity_threshold = self.get_parameter('similarity_threshold').value
        
        self.get_logger().info("Инициализация Voice Controller...")

    def execute_command(self, tools):
        """Выполнение команды на основе распознанных инструментов."""
        if not tools:
            return
            
        action = tools[0]
        
        if action == 'собачка':
            self.robot_utils.move_to_position(self.dog_pose, "Позиция собачки")
            self.robot_utils.pause_at_position("Позиция собачки", self.get_parameter('pause_duration').value)
            
        elif action == 'назад':
            self.robot_utils.move_to_position(self.waiting_pose, "Начальная позиция")
            
        elif action == 'посмотри':
            for i, pose in enumerate(self.looking_poses):
                self.robot_utils.move_to_position(pose, f"Позиция осмотра {i+1}")
                self.robot_utils.pause_at_position(f"Позиция осмотра {i+1}", 1.0)

    def run(self):
        """Основной цикл работы узла."""
        try:
            # Инициализация робота
            self.robot_utils.initialize_robot(self.robot_host)
            
            # Переход в начальную позицию
            self.get_logger().info("Переход в начальную позицию...")
            if not self.robot_utils.move_to_position(self.waiting_pose, "Начальная позиция"):
                return
            
            # Инициализация аудио
            model_path = os.path.join(
                get_package_share_directory('voice_controlled_robot'),
                'resources',
                'vosk-model-small-ru-0.22'
            )
            
            with self.audio_utils.initialize_audio(model_path, self.sample_rate) as stream:
                self.get_logger().info("🎤 Голосовой ассистент запущен. Говорите...")
                
                while rclpy.ok():
                    data = self.audio_utils.audio_queue.get()
                    if self.audio_utils.recognizer.AcceptWaveform(data):
                        result = json.loads(self.audio_utils.recognizer.Result())
                        text = result.get("text", "").strip()
                        
                        if text:
                            self.get_logger().info(f"🗣️ Команда: {text}")
                            
                            tools = self.speech_utils.find_tools_in_command(text, self.similarity_threshold)
                            
                            if tools:
                                self.get_logger().info(f"🎯 Распознаны действия: {', '.join(tools)}")
                                self.execute_command(tools)
                            else:
                                self.get_logger().warn("❌ Инструменты не распознаны")
                                
                    else:
                        partial_result = json.loads(self.audio_utils.recognizer.PartialResult())
                        partial_text = partial_result.get("partial", "").strip()
                        if partial_text:
                            print(f"▌ Слушаю: {partial_text}", end='\r')
                            
        except KeyboardInterrupt:
            self.get_logger().info("✅ Ассистент остановлен")
        except Exception as e:
            self.get_logger().error(f"❌ Ошибка: {e}")
        finally:
            if self.robot_utils.robot:
                self.robot_utils.robot.close()

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
