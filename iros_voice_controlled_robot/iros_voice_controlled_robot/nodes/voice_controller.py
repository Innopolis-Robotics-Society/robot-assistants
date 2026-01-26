#!/usr/bin/env python3

import json
import queue
import os
import sys
import yaml

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from vosk import Model, KaldiRecognizer
import sounddevice as sd

from ament_index_python.packages import get_package_share_directory

from iros_voice_controlled_robot.utils.audio_utils import AudioUtils
from iros_voice_controlled_robot.utils.grammar_utils import generate_grammar
from iros_voice_controlled_robot.utils.command_parser import match


class VoiceController(Node):

    def __init__(self):
        super().__init__('voice_controller')

        # Загрузка конфигурации
        pkg_path = get_package_share_directory('iros_voice_controlled_robot')
        cfg_path = os.path.join(pkg_path, 'config', 'tools_config.yaml')

        with open(cfg_path) as f:
            self.cfg = yaml.safe_load(f)

        self.wake_word = self.cfg.get('wake_word', '')

        # Путь к модели Vosk
        model_path = os.path.join(
            pkg_path,
            'models',
            'vosk-model-small-en-us'
        )

        self.get_logger().info(f'Loading Vosk model: {model_path}')
        self.model = Model(model_path)

        # Генерация грамматики
        grammar = generate_grammar(self.cfg)
        self.get_logger().info(f'Grammar size: {len(grammar)}')

        self.recognizer = KaldiRecognizer(
            self.model,
            16000,
            json.dumps(grammar)
        )

        # Очередь и поток для аудио
        self.audio_queue = queue.Queue()
        self.audio_device = AudioUtils.pick_input_device()

        # Проверка доступности аудио устройства
        if self.audio_device is None:
            self.get_logger().error('No valid audio device found.')
            return

        self.get_logger().info(f'Using audio device: {self.audio_device}')

        self.stream = sd.RawInputStream(
            callback=self.audio_callback,
            samplerate=16000,
            blocksize=8000,
            device=self.audio_device,
            dtype='int16',
            channels=1
        )
        self.stream.start()

        # Создание публикации
        self.pub = self.create_publisher(String, '/voice/command', 10)
        self.create_timer(0.1, self.process_audio)

        self.get_logger().info('Voice controller READY')

    def audio_callback(self, indata, frames, time, status):
        # Логирование состояния callback
        if status:
            self.get_logger().warn(f'Audio callback status: {status}')
        self.audio_queue.put(bytes(indata))
        #self.get_logger().info(f'Audio callback received {frames} frames')

    def process_audio(self):
        while not self.audio_queue.empty():
            data = self.audio_queue.get()

            # Логирование размера полученных данных
            #self.get_logger().info(f'Processing audio data: {len(data)} bytes')

            if self.recognizer.AcceptWaveform(data):
                result = json.loads(self.recognizer.Result())
                self.get_logger().info(f'Vosk recognition result: {result}')
                text = result.get('text', '')

                if not text:
                    self.get_logger().info('No text recognized')
                    return

                # Проверка на wake word
                if self.wake_word and self.wake_word not in text:
                    self.get_logger().info(f'Wake word "{self.wake_word}" not found in: {text}')
                    return

                # Разбор команды
                parsed = match(text, self.cfg)
                if not parsed:
                    self.get_logger().info(f'Command parsing failed for text: {text}')
                    return

                # Публикация результата
                msg = String()
                msg.data = json.dumps(parsed)
                self.pub.publish(msg)

                self.get_logger().info(f'Command: {parsed}')


def main():
    rclpy.init()
    node = VoiceController()
    if node:
        rclpy.spin(node)
    rclpy.shutdown()

