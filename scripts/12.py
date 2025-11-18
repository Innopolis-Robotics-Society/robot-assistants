import math
import queue
import re
import sys
import time
from difflib import SequenceMatcher
import json

import sounddevice as sd
import vosk
import URBasic
import URBasic.robotModel
import URBasic.urScriptExt


class VoiceControlledRobot:
    """Класс для управления роботом с помощью голосовых команд."""
    
    # Константы для подключения к роботу
    HOST = "192.168.0.100"
    RAW = 2.857
    PITCH = -1.309
    YAW = 0.0
    
    # Позиции робота
    WAITING_POSE = [-0.19, -0.315, 0.350, 2.783, -1.353, 0.03]
    LOOKING_POSE = [
        [-0.22, -0.873, 1, 1.539, -0.749, 0.584],
        [0.185, -0.893, 1.1, 1.614, -0.831, 0.133],
        [-0.610, -0.650, 0.890, 1.607, -0.822, 0.750],
        [-0.228, -0.388, 1.26, 1.435, -0.730, 0.592],
        [-0.22, -0.873, 1, 1.539, -0.749, 0.584]
    ]
    HAMMER_POSE = [
        [-0.179, -1.036, 0.201, 2.853, -1.312, 0.106],
        [0.240, -1.025, 0.739, 2.007, -0.427, 0.861]
    ]
    DOG_POSE = [-0.828, -0.624, 0.490, 2.003, -1.873, 0.08]
    
    # Словарь инструментов
    TOOLS_DICT = {
        'посмотри': ['посмотри', 'изучи', 'осмотри', 'взглянги'],
        'собачка': ['собака', 'собаку', 'собачку'],
        'назад': ['вернись', 'начало', 'назад']
    }

    def __init__(self):
        """Инициализация голосового ассистента и робота."""
        self.robot = None
        self.robot_model = None
        self.audio_queue = queue.Queue()
        self.sample_rate = None
        self.audio_device = None
        self.recognizer = None
        
    @staticmethod
    def normalize_angle(angle):
        """Нормализует угол в диапазон [-pi, pi].
        
        Args:
            angle (float): Исходный угол
            
        Returns:
            float: Нормализованный угол
        """
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle

    def check_position(self, current_pos, target_pos, linear_tolerance=0.001, angle_tolerance=0.01):
        """Проверяет, достигнута ли целевая позиция.
        
        Args:
            current_pos (list): Текущая позиция [x, y, z, rx, ry, rz]
            target_pos (list): Целевая позиция [x, y, z, rx, ry, rz]
            linear_tolerance (float): Допуск для линейных координат
            angle_tolerance (float): Допуск для углов
            
        Returns:
            bool: True если позиция достигнута, иначе False
        """
        position_reached = True
        
        print("Проверка позиции:")
        for i, (current, target) in enumerate(zip(current_pos, target_pos)):
            if i < 3:  # Позиция (x, y, z) - линейные координаты
                diff = abs(current - target)
                if diff > linear_tolerance:
                    position_reached = False
                status = '✓' if diff <= linear_tolerance else '✗'
                print(f"Ось {i}(лин.): текущая={current:08.6f}, "
                      f"целевая={target:08.6f}, разница={diff:08.6f} {status}")
            else:  # Ориентация (rx, ry, rz) - углы
                norm_current = self.normalize_angle(current)
                norm_target = self.normalize_angle(target)
                diff = abs(-norm_current - norm_target)
                if diff > angle_tolerance:
                    position_reached = False
                status = '✓' if diff <= angle_tolerance else '✗'
                print(f"Ось {i}(угл.): текущая={current:08.6f}→{norm_current:08.6f}, "
                      f"целевая={target:08.6f}→{norm_target:08.6f}, "
                      f"разница={diff:08.6f} {status}")
        
        print(f"Позиция достигнута: {position_reached}")
        print("-" * 80)
        
        return position_reached

    def wait_for_position(self, target_pos, position_name="", timeout=15.0, check_interval=0.05):
        """Ожидает достижения целевой позиции с таймаутом.
        
        Args:
            target_pos (list): Целевая позиция
            position_name (str): Название позиции для логов
            timeout (float): Максимальное время ожидания
            check_interval (float): Интервал проверки
            
        Returns:
            bool: True если позиция достигнута, иначе False
        """
        start_time = time.time()
        attempt = 0
        
        while time.time() - start_time < timeout:
            attempt += 1
            current_pos = self.robot.get_actual_tcp_pose_custom()
            print(f"Попытка {attempt} для {position_name}:")
            
            if self.check_position(current_pos, target_pos):
                print(f"✓ Позиция '{position_name}' успешно достигнута!")
                return True
            
            # Показываем прогресс каждые 10 попыток
            if attempt % 10 == 0:
                elapsed = time.time() - start_time
                print(f"Прогресс: {elapsed:.1f}с из {timeout}с")
            
            time.sleep(check_interval)
        
        print(f"✗ Таймаут: позиция '{position_name}' не достигнута за {timeout} секунд")
        return False

    @staticmethod
    def pause_at_position(position_name, duration=2.0):
        """Пауза в достигнутой позиции.
        
        Args:
            position_name (str): Название позиции
            duration (float): Длительность паузы
        """
        print(f"⏸️  Пауза {duration} секунд в позиции '{position_name}'...")
        time.sleep(duration)
        print(f"▶️  Пауза завершена, продолжаем работу.")

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
        """Поиск инструментов в голосовой команде.
        
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

    def pick_input_device(self):
        """Выбор аудио устройства для записи.
        
        Returns:
            int: ID выбранного устройства
            
        Raises:
            RuntimeError: Если нет подходящих аудио устройств
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

    def audio_callback(self, indata, frames, time_info, status):
        """Callback функция для обработки аудио потока.
        
        Args:
            indata: Входные аудио данные
            frames: Количество кадров
            time_info: Временная информация
            status: Статус аудио потока
        """
        if status:
            print(status, file=sys.stderr)
        self.audio_queue.put(bytes(indata))

    def initialize_robot(self):
        """Инициализация соединения с роботом."""
        print("Initialization UR")
        self.robot_model = URBasic.robotModel.RobotModel()
        self.robot = URBasic.urScriptExt.UrScriptExt(
            host=self.HOST, 
            robotModel=self.robot_model
        )
        self.robot.init_realtime_control()
        
        # Получение текущей позиции
        current_pos = self.robot.get_actual_tcp_pose_custom()
        print('Текущая позиция робота: '
              '[{: 08.6f}, {: 08.6f}, {: 08.6f}, '
              '{: 08.6f}, {: 08.6f}, {: 08.6f}]'.format(*current_pos))

    def move_to_position(self, target_pos, position_name):
        """Перемещение робота в указанную позицию.
        
        Args:
            target_pos: Целевая позиция
            position_name: Название позиции для логов
        """
        print(f"Переход в позицию '{position_name}': {target_pos}")
        self.robot.set_realtime_pose(target_pos)
        
        if not self.wait_for_position(target_pos, position_name):
            print(f"Ошибка: не удалось достичь позиции '{position_name}'")
            self.robot.close()
            sys.exit(1)

    def execute_command(self, tools):
        """Выполнение команды на основе распознанных инструментов.
        
        Args:
            tools (list): Список распознанных инструментов/действий
        """
        if not tools:
            return
            
        action = tools[0]
        
        if action == 'собачка':
            self.move_to_position(self.DOG_POSE, "Позиция собачки")
            
        elif action == 'назад':
            self.move_to_position(self.WAITING_POSE, "Начальная позиция")
            
        elif action == 'посмотри':
            for i, pose in enumerate(self.LOOKING_POSE):
                self.move_to_position(pose, f"Позиция осмотра {i+1}")
                
        # Добавьте здесь обработку других команд

    def voice_assistant(self):
        """Основной цикл голосового ассистента."""
        try:
            # Инициализация робота
            self.initialize_robot()
            
            input('Нажмите Enter для перехода в начальную позицию...')
            
            # Переход в начальную позицию
            self.move_to_position(self.WAITING_POSE, "Начальная позиция")

            # Инициализация аудио системы
            self.audio_device = self.pick_input_device()
            self.sample_rate = int(sd.query_devices(
                self.audio_device, 'input')["default_samplerate"]
            )
            
            model = vosk.Model("model_stt/vosk-model-small-ru-0.22")
            self.recognizer = vosk.KaldiRecognizer(model, self.sample_rate)
            
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
                                print(f"🎯 Распознаны действия: {', '.join(tools)}")
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
        finally:
            if self.robot:
                self.robot.close()

    def run(self):
        """Запуск голосового ассистента."""
        self.voice_assistant()


if __name__ == "__main__":
    assistant = VoiceControlledRobot()
    assistant.run()