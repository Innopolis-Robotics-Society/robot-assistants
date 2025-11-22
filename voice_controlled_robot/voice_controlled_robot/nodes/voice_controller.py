#!/usr/bin/env python3

import json
import queue
import os
import re
import sys

import rclpy
from rclpy.node import Node

from std_msgs.msg import String
from vosk import Model, KaldiRecognizer
import sounddevice as sd

from voice_controlled_robot.utils.audio_utils import AudioUtils


class VoiceController(Node):
    """Нода: слушает микрофон, распознаёт речь, выделяет команду после ключевого слова и
    публикует только команды из разрешённого списка.
    """

    def __init__(self):
        super().__init__('voice_controller')

        # Параметры
        self.declare_parameters(
            namespace='',
            parameters=[
                ('model_path', ''),
                ('audio_device', 'auto'),
                ('sample_rate', 16000),
                ('enable_partial_results', True),
                # Ключевое слово (обращение к роботу), напр. "робот"
                ('wake_word', 'робот'),
            ]
        )

        model_path_param = self.get_parameter('model_path').value
        self.audio_device_param = self.get_parameter('audio_device').value
        self.sample_rate = int(self.get_parameter('sample_rate').value)
        self.enable_partial_results = bool(self.get_parameter('enable_partial_results').value)
        self.wake_word = self.get_parameter('wake_word').value.strip().lower()

        # Разрешённые команды и их варианты в тексте
        # Публикуется КАНОНИЧЕСКИЙ ключ: "вперед", "назад" и т.п.
        self.allowed_commands = {
            'вперед': ['вперед', 'вперёд', 'прямо'],
            'назад': ['назад', 'обратно'],
            'влево': ['влево', 'налево'],
            'вправо': ['вправо', 'направо'],
            'стоп': ['стоп', 'остановись', 'стой'],
        }

        # Определение пути к модели
        self.model_path = self._resolve_model_path(model_path_param)
        self.get_logger().info(f'Используется модель: {self.model_path}')

        # Публикаторы
        # Полный распознанный текст
        self.recognized_speech_pub = self.create_publisher(String, 'voice/recognized_speech', 10)
        # Только команда (из разрешённого списка), выделенная после wake_word
        self.command_pub = self.create_publisher(String, 'voice/command', 10)
        # Отладочная информация
        self.debug_pub = self.create_publisher(String, 'voice/debug', 10)

        # Очередь для аудиоданных
        self.audio_queue = queue.Queue()

        self.audio_stream = None
        self.model = None
        self.recognizer = None

        # Инициализация модели и аудио
        self.initialize_recognizer_and_audio()

        # Таймер обработки аудио
        self.create_timer(0.1, self.process_audio)

        self.get_logger().info('Нода голосового управления запущена')

    def _resolve_model_path(self, model_path_param: str) -> str:
        """Определение пути к модели Vosk."""
        if model_path_param:
            if not os.path.exists(model_path_param):
                self.get_logger().error(f'Указанный model_path не существует: {model_path_param}')
                raise FileNotFoundError(model_path_param)
            return model_path_param

        possible_paths = [
            os.path.expanduser('~/vosk-models/vosk-model-small-ru-0.22'),
            '/usr/share/vosk-models/vosk-model-small-ru-0.22',
            os.path.join(
                os.path.dirname(__file__),
                '../../../../share/voice_controlled_robot/models/vosk-model-small-ru-0.22'
            ),
        ]

        for path in possible_paths:
            if os.path.exists(path):
                return path

        self.get_logger().error('Модель Vosk не найдена. Укажите параметр model_path или установите модель в одно из стандартных мест.')
        raise FileNotFoundError('Модель Vosk не найдена')

    def initialize_recognizer_and_audio(self):
        """Инициализация модели Vosk и аудиопотока."""
        try:
            self.get_logger().info(f'Загрузка модели Vosk из: {self.model_path}')
            self.model = Model(self.model_path)
            self.recognizer = KaldiRecognizer(self.model, self.sample_rate)
        except Exception as e:
            self.get_logger().error(f'Ошибка загрузки модели Vosk: {e}')
            raise

        try:
            if self.audio_device_param == 'auto':
                self.audio_device = AudioUtils.pick_input_device()
            else:
                self.audio_device = int(self.audio_device_param)

            self.audio_stream = sd.RawInputStream(
                callback=self.audio_callback,
                channels=1,
                samplerate=self.sample_rate,
                device=self.audio_device,
                dtype='int16'
            )
            self.audio_stream.start()

            self.get_logger().info(f'Аудио инициализировано, устройство: {self.audio_device}')
        except Exception as e:
            self.get_logger().error(f'Ошибка инициализации аудио: {e}')
            raise

    def audio_callback(self, indata, frames, time, status):
        """Callback для sounddevice: складывает сырые данные в очередь."""
        if status:
            msg = f'Аудио статус: {status}'
            self.get_logger().warn(msg)
            dbg = String()
            dbg.data = msg
            self.debug_pub.publish(dbg)

        self.audio_queue.put(bytes(indata))

    def extract_text_after_wake_word(self, text: str):
        """
        Возвращает текст после ключевого слова (wake_word).

        Если wake_word пустой — возвращает исходный текст.
        Если ключевое слово не найдено — возвращает None.
        """
        raw_text = text.strip()
        if not raw_text:
            return None

        if not self.wake_word:
            return raw_text

        lower_text = raw_text.lower()
        idx = lower_text.find(self.wake_word)
        if idx == -1:
            return None

        start_idx = idx + len(self.wake_word)
        command_part = raw_text[start_idx:].strip()

        # Убираем начальные разделители/пробелы
        command_part = re.sub(r'^[,.:;«»"\s]+', '', command_part).strip()

        return command_part or None

    def detect_allowed_command(self, text_after_wake: str):
        """
        Находит разрешённую команду в тексте после wake_word.

        Возвращает каноническое имя команды (ключ словаря self.allowed_commands) или None.
        """
        if not text_after_wake:
            return None

        lower_cmd = text_after_wake.lower()

        for canonical, variants in self.allowed_commands.items():
            for phrase in variants:
                # Ищем фразу как отдельное слово или часть, окружённую границами слова
                pattern = r'\b' + re.escape(phrase) + r'\b'
                if re.search(pattern, lower_cmd):
                    return canonical

        return None

    def process_audio(self):
        """Обработка аудиоданных и распознавание речи."""
        if self.recognizer is None:
            return

        try:
            while not self.audio_queue.empty():
                data = self.audio_queue.get()

                if self.recognizer.AcceptWaveform(data):
                    result = json.loads(self.recognizer.Result())
                    text = result.get("text", "").strip()
                    if not text:
                        continue

                    # Публикуем полный распознанный текст
                    speech_msg = String()
                    speech_msg.data = text
                    self.recognized_speech_pub.publish(speech_msg)
                    self.get_logger().info(f'Распознано: "{text}"')

                    # Текст после ключевого слова
                    tail = self.extract_text_after_wake_word(text)
                    if tail is None:
                        dbg_msg = String()
                        dbg_msg.data = f'wake_word="{self.wake_word}" не найден, текст="{text}"'
                        self.debug_pub.publish(dbg_msg)
                        continue

                    # Поиск разрешённой команды
                    command = self.detect_allowed_command(tail)
                    if command is not None:
                        cmd_msg = String()
                        cmd_msg.data = command
                        self.command_pub.publish(cmd_msg)

                        dbg_msg = String()
                        dbg_msg.data = f'Команда распознана: canonical="{command}", tail="{tail}"'
                        self.debug_pub.publish(dbg_msg)
                        self.get_logger().info(f'Команда: "{command}"')
                    else:
                        dbg_msg = String()
                        dbg_msg.data = f'Команда не распознана в tail="{tail}"'
                        self.debug_pub.publish(dbg_msg)

                elif self.enable_partial_results:
                    partial_result = json.loads(self.recognizer.PartialResult())
                    partial_text = partial_result.get("partial", "").strip()
                    if partial_text:
                        dbg_msg = String()
                        dbg_msg.data = f'partial="{partial_text}"'
                        self.debug_pub.publish(dbg_msg)

        except Exception as e:
            self.get_logger().error(f'Ошибка обработки аудио: {e}')
            dbg_msg = String()
            dbg_msg.data = f'Ошибка обработки аудио: {e}'
            self.debug_pub.publish(dbg_msg)

    def destroy_node(self):
        """Освобождение ресурсов."""
        if self.audio_stream is not None:
            try:
                self.audio_stream.stop()
                self.audio_stream.close()
            except Exception:
                pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)

    voice_controller = None
    try:
        voice_controller = VoiceController()
        rclpy.spin(voice_controller)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f'Ошибка: {e}', file=sys.stderr)
    finally:
        if voice_controller is not None:
            voice_controller.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
