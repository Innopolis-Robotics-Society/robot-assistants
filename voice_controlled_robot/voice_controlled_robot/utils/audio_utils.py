import sounddevice as sd
import sys
from difflib import SequenceMatcher
import os

class AudioUtils:
    """Утилиты для работы с аудио."""
    
    @staticmethod
    def similarity(a, b):
        """Вычисляет схожесть двух строк."""
        return SequenceMatcher(None, a, b).ratio()

    @staticmethod
    def pick_input_device(preferred_devices=None):
        """Выбор аудиоустройства для записи."""
        if preferred_devices is None:
            preferred_devices = ['pulse', 'default', 'alsa']
            
        try:
            devices = sd.query_devices()
            print("Доступные аудиоустройства:")
            for i, device in enumerate(devices):
                if device["max_input_channels"] > 0:
                    print(f"  {i}: {device['name']} (входные каналы: {device['max_input_channels']})")
            
            for preferred in preferred_devices:
                for i, device in enumerate(devices):
                    if (device["max_input_channels"] > 0 and 
                        preferred in device["name"].lower()):
                        print(f"Выбрано устройство: {i} - {device['name']}")
                        return i
            
            # Если не нашли по предпочтениям, берем первое доступное
            for i, device in enumerate(devices):
                if (device["max_input_channels"] > 0 and 
                    "hdmi" not in device["name"].lower() and
                    "output" not in device["name"].lower()):
                    print(f"Автовыбор устройства: {i} - {device['name']}")
                    return i
                    
            raise RuntimeError("Нет подходящих входных аудиоустройств")
            
        except Exception as e:
            print(f"Ошибка при выборе аудиоустройства: {e}")
            raise

    @staticmethod
    def list_audio_devices():
        """Список всех аудиоустройств."""
        devices = sd.query_devices()
        input_devices = []
        
        print("\n=== Доступные аудиоустройства ===")
        for i, device in enumerate(devices):
            if device["max_input_channels"] > 0:
                status = "✓ ВХОД"
                input_devices.append((i, device))
            elif device["max_output_channels"] > 0:
                status = "ВЫХОД"
            else:
                status = "N/A"
                
            print(f"{i:2d}: {device['name']} [{status}]")
            print(f"     Sample rate: {device['default_samplerate']} Hz")
            print(f"     Channels: in={device['max_input_channels']}, out={device['max_output_channels']}")
        
        return input_devices
