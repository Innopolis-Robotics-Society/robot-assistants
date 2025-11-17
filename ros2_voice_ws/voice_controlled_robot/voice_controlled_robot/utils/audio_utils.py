import queue
import sys
import sounddevice as sd
import json
import vosk
from rclpy.node import Node

class AudioUtils:
    """Утилиты для работы с аудио."""
    
    def __init__(self, node: Node):
        self.node = node
        self.audio_queue = queue.Queue()
        self.sample_rate = None
        self.audio_device = None
        self.recognizer = None
        
    def pick_input_device(self):
        """Выбор аудио устройства для записи."""
        devices = sd.query_devices()
        
        # Приоритет: pulse → default → любой с входом
        for i, device in enumerate(devices):
            if (device["max_input_channels"] > 0 and 
                device["name"].lower() == "pulse"):
                self.node.get_logger().info(f"Выбрано аудио устройство: {device['name']}")
                return i
                
        for i, device in enumerate(devices):
            if (device["max_input_channels"] > 0 and 
                device["name"].lower() == "default"):
                self.node.get_logger().info(f"Выбрано аудио устройство: {device['name']}")
                return i
                
        for i, device in enumerate(devices):
            if (device["max_input_channels"] > 0 and 
                "hdmi" not in device["name"].lower()):
                self.node.get_logger().info(f"Выбрано аудио устройство: {device['name']}")
                return i
                
        raise RuntimeError("Нет входных аудиоустройств")

    def audio_callback(self, indata, frames, time_info, status):
        """Callback функция для обработки аудио потока."""
        if status:
            self.node.get_logger().warn(f"Аудио статус: {status}")
        self.audio_queue.put(bytes(indata))

    def initialize_audio(self, model_path, sample_rate):
        """Инициализация аудио системы."""
        self.sample_rate = sample_rate
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
