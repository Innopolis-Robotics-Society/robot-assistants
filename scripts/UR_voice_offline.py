import sounddevice as sd
import vosk
import json
import queue
import sys
import re
from difflib import SequenceMatcher


class VoiceControlledRobotOffline:
    """Класс для управления роботом с помощью голосовых команд."""

    # Словарь инструментов
    TOOLS_DICT = {
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

    def __init__(self, model_path="model_stt/vosk-model-small-ru-0.22"):
        """Инициализация голосового ассистента.
        
        Args:
            model_path (str): Путь к модели распознавания речи
        """
        self.audio_queue = queue.Queue()
        self.sample_rate = None
        self.audio_device = None
        self.recognizer = None
        self.model_path = model_path
        
    @staticmethod    
    def similarity(a, b):
        """Вычисляет схожесть двух строк.
        
        Args:
            a (str): Первая строка
            b (str): Вторая строка
            
        Returns:
            float: Коэффициент схожести (0-1)
        """
        return SequenceMatcher(None, a, b).ratio()

    def find_tools_in_command(self, command, similarity_threshold=0.7):
        """Поиск инструментов в команде.
        
        Args:
            command (str): Распознанная голосовая команда
            similarity_threshold (float): Порог схожести для нечеткого поиска
            
        Returns:
            list: Список найденных инструментов
        """
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

    def audio_callback(self, indata, frames, time, status):
        """Callback-функция для обработки аудиопотока.
        
        Args:
            indata: Входные аудиоданные
            frames: Количество кадров
            time: Временная информация
            status: Статус аудиопотока
        """
        if status:
            print(status, file=sys.stderr)
        self.audio_queue.put(bytes(indata))

    def pick_input_device(self):
        """Выбор аудиоустройства для записи.
        
        Returns:
            int: ID выбранного устройства
            
        Raises:
            RuntimeError: Если нет подходящих аудиоустройств
        """
        devices = sd.query_devices()
        
        # Приоритет: pulse → default → любой с входом
        for i, device in enumerate(devices):
            if (device["max_input_channels"] > 0 and 
                device["name"].lower() == "pulse"):
                return i
                
        for i, device in enumerate(devices):
            if (device["max_input_channels"] > 0 and 
                device["name"].lower() == "default"):
                return i
                
        for i, device in enumerate(devices):
            if (device["max_input_channels"] > 0 and 
                "hdmi" not in device["name"].lower()):
                return i
                
        raise RuntimeError("Нет входных аудиоустройств")

    def initialize_audio(self):
        """Инициализация аудиосистемы."""
        self.audio_device = self.pick_input_device()
        self.sample_rate = int(sd.query_devices(
            self.audio_device, 'input')["default_samplerate"]
        )
        
        # Загрузка модели распознавания речи
        model = vosk.Model(self.model_path)
        self.recognizer = vosk.KaldiRecognizer(model, self.sample_rate)

    def execute_command(self, tools):
        """Выполнение команды на основе распознанных инструментов.
        
        Args:
            tools (list): Список распознанных инструментов
        """
        if tools:
            print(f"🎯 Выполняю команду для инструментов: {', '.join(tools)}")
            # Здесь можно добавить логику управления роботом
            # Например: self.robot.move_to_tool_position(tools[0])
        else:
            print("❌ Неизвестная команда")

    def voice_assistant(self):
        """Основной цикл голосового ассистента."""
        try:
            self.initialize_audio()
            
            with sd.RawInputStream(
                callback=self.audio_callback,
                channels=1,
                samplerate=self.sample_rate,
                device=self.audio_device,
                dtype='int16'
            ):
                print("🎤 Голосовой ассистент запущен. Говорите...")
                
                while True:
                    data = self.audio_queue.get()
                    if self.recognizer.AcceptWaveform(data):
                        result = json.loads(self.recognizer.Result())
                        text = result.get("text", "").strip()
                        
                        if text:
                            print(f"\n🗣️ Команда: {text}")
                            
                            # Ищем инструменты в команде
                            tools = self.find_tools_in_command(text)
                            
                            if tools:
                                print(f"🎯 Распознаны инструменты: {', '.join(tools)}")
                                self.execute_command(tools)
                            else:
                                print("❌ Инструменты не распознаны")
                                
                    else:
                        # Вывод частичных результатов
                        partial_result = json.loads(self.recognizer.PartialResult())
                        partial_text = partial_result.get("partial", "").strip()
                        if partial_text:
                            print(f"▌ Слушаю: {partial_text}", end='\r')
                            
        except KeyboardInterrupt:
            print("\n✅ Ассистент остановлен")
        except Exception as e:
            print(f"❌ Ошибка: {e}")

    def run(self):
        """Запуск голосового ассистента."""
        self.voice_assistant()


if __name__ == "__main__":
    assistant = VoiceControlledRobotOffline()
    assistant.run()