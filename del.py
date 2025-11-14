import sounddevice as sd
import vosk # silero, s
import json
import queue
import sys
import re
import time
import math
from difflib import SequenceMatcher

import URBasic
import URBasic.robotModel
import URBasic.urScriptExt

RAW = 2.857
PITCH = -1.309
YAW = 0.0
HOST = "192.168.0.100"

# Словарь инструментов (можно вынести в отдельный файл)
tools_dict = {
    'посмотри': ['посмотри', 'изучи', 'осмотри', 'взглянги'],
    'собачка': ['собака', 'собаку', 'собачку'],
    'назад': ['вернись', 'начало', 'назад']
}

waiting_pose = [-0.19, -0.315, 0.350, 2.783, -1.353, 0.03]
looking_pose = [[-0.22, -0.873, 1, 1.539, -0.749, 0.584], 
                [0.185, -0.893, 1.1, 1.614, -0.831, 0.133],
                [-0.610, -0.650, 0.890, 1.607, -0.822, 0.750],
                [-0.228, -0.388, 1.26, 1.435, -0.730, 0.592],
                [-0.22, -0.873, 1, 1.539, -0.749, 0.584]]
hammer_pose  = [[-0.179, -1.036, 0.201, 2.853, -1.312, 0.106], 
                [0.240, -1.025, 0.739, 2.007, -0.427, 0.861]]
dog_pose     =  [-0.828, -0.624, 0.490, 2.003, -1.873, 0.08]

def normalize_angle(angle):
    """Нормализует угол в диапазон [-pi, pi]"""
    while angle > math.pi:
        angle -= 2 * math.pi
    while angle < -math.pi:
        angle += 2 * math.pi
    return angle

def check_position(current_pos, target_pos, linear_tolerance=0.001, angle_tolerance=100.01):
    """Проверяет, достигнута ли целевая позиция с учетом нормализации углов"""
    position_reached = True
    
    print("Проверка позиции:")
    for i, (current, target) in enumerate(zip(current_pos, target_pos)):
        if i < 3:  # Позиция (x, y, z) - линейные координаты
            diff = abs(current - target)
            if diff > linear_tolerance:
                position_reached = False
            print(f"Ось {i}(лин.): текущая={current:08.6f}, целевая={target:08.6f}, разница={diff:08.6f} {'✓' if diff <= linear_tolerance else '✗'}")
        else:  # Ориентация (rx, ry, rz) - углы
            # Нормализуем углы перед сравнением
            norm_current = normalize_angle(current)
            norm_target = normalize_angle(target)
            diff = abs(-norm_current - norm_target)
            # print(f"norm_targ:= {norm_target}" )
            # print(f"norm_cur:= {current}" )
            if diff > angle_tolerance:
                position_reached = False
            print(f"Ось {i}(угл.): текущая={current:08.6f}→{norm_current:08.6f}, целевая={target:08.6f}→{norm_target:08.6f}, разница={diff:08.6f} {'✓' if diff <= angle_tolerance else '✗'}")
    
    print(f"Позиция достигнута: {position_reached}")
    print("-" * 80)
    
    return position_reached

def wait_for_position(robot, target_pos, position_name="", timeout=15.0, check_interval=0.05):
    """Ожидает достижения целевой позиции с таймаутом"""
    start_time = time.time()
    attempt = 0
    
    while time.time() - start_time < timeout:
        attempt += 1
        current_pos = robot.get_actual_tcp_pose_custom()
        print(f"Попытка {attempt} для {position_name}:")
        
        if check_position(current_pos, target_pos):
            print(f"✓ Позиция '{position_name}' успешно достигнута!")
            return True
        
        # Показываем прогресс каждые 10 попыток
        if attempt % 10 == 0:
            elapsed = time.time() - start_time
            print(f"Прогресс: {elapsed:.1f}с из {timeout}с")
        
        time.sleep(check_interval)
    
    print(f"✗ Таймаут: позиция '{position_name}' не достигнута за {timeout} секунд")
    return False

def pause_at_position(position_name, duration=2.0):
    """Пауза в достигнутой позиции"""
    print(f"⏸️  Пауза {duration} секунд в позиции '{position_name}'...")
    time.sleep(duration)
    print(f"▶️  Пауза завершена, продолжаем работу.")


def similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()

def find_tools_in_command(command, similarity_threshold=0.7):
    """Поиск инструментов в команде"""
    command = command.lower().strip()
    found_tools = []
    
    # Быстрый поиск по подстроке
    for tool, keywords in tools_dict.items():
        for keyword in keywords:
            if keyword in command:
                found_tools.append(tool)
                break
    
    if found_tools:
        return list(set(found_tools))
    
    # Поиск по схожести для случаев с ошибками
    words = re.findall(r'\w+', command)
    for word in words:
        for tool, keywords in tools_dict.items():
            for keyword in keywords:
                if similarity(word, keyword) >= similarity_threshold:
                    found_tools.append(tool)
                    break
    
    return list(set(found_tools))

def pick_input_device():
    devs = sd.query_devices()
    # приоритет: pulse → default → любой с входом
    for i,d in enumerate(devs):
        if d["max_input_channels"] > 0 and d["name"].lower() == "pulse":
            return i
    for i,d in enumerate(devs):
        if d["max_input_channels"] > 0 and d["name"].lower() == "default":
            return i
    for i,d in enumerate(devs):
        if d["max_input_channels"] > 0 and "hdmi" not in d["name"].lower():
            return i
    raise RuntimeError("Нет входных аудиоустройств")

device_m = pick_input_device()
samplerate = int(sd.query_devices(device_m, 'input')["default_samplerate"])
q = queue.Queue()
model = vosk.Model("model_stt/vosk-model-small-ru-0.22")


def q_callback(indata, frames, time, status):
    if status:
        print(status, file=sys.stderr)
    q.put(bytes(indata))

def voice_assistant():
    try:

        # Инициализация робота
        #robotModel = URBasic.robotModel.RobotModel()
        print("Initialization UR")

        #robot = URBasic.urScriptExt.UrScriptExt(host=HOST, robotModel=robotModel)
        #robot.init_realtime_control()
        
        # Получение текущей позиции
        #current_pos = robot.get_actual_tcp_pose_custom()
        #print('Текущая позиция робота: [{: 08.6f}, {: 08.6f}, {: 08.6f}, {: 08.6f}, {: 08.6f}, {: 08.6f}]'.format(*current_pos))

        input('Нажмите Enter для перехода в начальную позицию...')


        # Переход в начальную позицию
        target_pos = waiting_pose  # Исправлено с 0.9 на 0.7
        print(f"Переход в начальную позицию: {target_pos}")
        #robot.set_realtime_pose(target_pos)
        
        # Ожидание достижения начальной позиции
        #if not wait_for_position(robot, target_pos, "Начальная позиция"):
        #    print("Ошибка: не удалось достичь начальной позиции")
        #    robot.close()
        #    exit()

        

        with sd.RawInputStream(callback=q_callback, 
                             channels=1, 
                             samplerate=samplerate, 
                             device=device_m, 
                             dtype='int16'):
            
            print("🎤 Голосовой ассистент запущен. Говорите...")
            
            rec = vosk.KaldiRecognizer(model, samplerate)
            
            while True:
                data = q.get()
                if rec.AcceptWaveform(data):
                    result = json.loads(rec.Result())
                    text = result.get("text", "").strip()
                    
                    if text:
                        print(f"\n🗣️ Команда: {text}")
                        
                        # Ищем инструменты в команде
                        tools = find_tools_in_command(text)
                        
                        if tools:
                            print(f"🎯 Распознаны действия: {', '.join(tools)}")
                            if tools[0] == 'собачка':
                                # Переход в начальную позицию
                                target_pos = dog_pose  # Исправлено с 0.9 на 0.7
                                print(f"Переход в начальную позицию: {target_pos}")
                                robot.set_realtime_pose(target_pos)
                                
                                # Ожидание достижения начальной позиции
                                if not wait_for_position(robot, target_pos, "Начальная позиция"):
                                    print("Ошибка: не удалось достичь начальной позиции")
                                    robot.close()
                                    exit()
                            if tools[0] == 'назад':
                                # Переход в начальную позицию
                                target_pos = waiting_pose  # Исправлено с 0.9 на 0.7
                                print(f"Переход в начальную позицию: {target_pos}")
                                robot.set_realtime_pose(target_pos)
                                
                                # Ожидание достижения начальной позиции
                                if not wait_for_position(robot, target_pos, "Начальная позиция"):
                                    print("Ошибка: не удалось достичь начальной позиции")
                                    robot.close()
                                    exit()
                            if tools[0] == 'молfgfghgк':
                                for pose in hammer_pose:
                                    # Переход в начальную позицию
                                    target_pos = pose  # Исправлено с 0.9 на 0.7
                                    print(f"Переход в начальную позицию: {target_pos}")
                                    robot.set_realtime_pose(target_pos)
                                    
                                    # Ожидание достижения начальной позиции
                                    if not wait_for_position(robot, target_pos, "Начальная позиция"):
                                        print("Ошибка: не удалось достичь начальной позиции")
                                        robot.close()
                                        exit()
                            if tools[0] == 'посмотри':
                                for pose in looking_pose:
                                    # Переход в начальную позицию
                                    target_pos = pose  # Исправлено с 0.9 на 0.7
                                    print(f"Переход в начальную позицию: {target_pos}")
                                    robot.set_realtime_pose(target_pos)
                                    
                                    # Ожидание достижения начальной позиции
                                    if not wait_for_position(robot, target_pos, "Начальная позиция"):
                                        print("Ошибка: не удалось достичь начальной позиции")
                                        robot.close()
                                        exit()
                        else:
                            print("❌ Инструменты не распознаны")
                            
                else:
                    # Вывод частичных результатов (опционально)
                    partial_result = json.loads(rec.PartialResult())
                    partial_text = partial_result.get("partial", "").strip()
                    if partial_text:
                        print(f"▌ Слушаю: {partial_text}", end='\r')
                        
    except KeyboardInterrupt:
        print("\n✅ Ассистент остановлен")
    except Exception as e:
        print(f"❌ Ошибка: {e}")
    finally:
        if 'robot' in locals():
            robot.close()


if __name__ == "__main__":
    voice_assistant()
    
