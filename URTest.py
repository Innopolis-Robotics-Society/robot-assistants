import URBasic
import URBasic.robotModel
import URBasic.urScriptExt
import time
import math
import sys, json
import sys, os
from pathlib import Path
# sys.path.append("~/take/ur10-assistant/hoba")
VPATH = Path.home() / "take" / "ur10-assistant" / "hoba"
assert VPATH.exists(), f"bad path: {VPATH}"

# prepend once
p = str(VPATH)
if p not in sys.path:
    sys.path.insert(0, p)

# import ONE function, with a clear alias
from localizer import measure_xy_once as measure_xy


HOST = "192.168.0.100"

def normalize_angle(angle):
    """Нормализует угол в диапазон [-pi, pi]"""
    while angle > math.pi:
        angle -= 2 * math.pi
    while angle < -math.pi:
        angle += 2 * math.pi
    return angle

def check_position(current_pos, target_pos, linear_tolerance=0.001, angle_tolerance=0.001):
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
            diff = abs(norm_current - norm_target)
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

# Инициализация робота
robotModel = URBasic.robotModel.RobotModel()
print("Initialization UR")

try:
    robot = URBasic.urScriptExt.UrScriptExt(host=HOST, robotModel=robotModel)
    robot.init_realtime_control()
    
    # Получение текущей позиции
    current_pos = robot.get_actual_tcp_pose_custom()
    print('Текущая позиция робота: [{: 08.6f}, {: 08.6f}, {: 08.6f}, {: 08.6f}, {: 08.6f}, {: 08.6f}]'.format(*current_pos))

    input('Нажмите Enter для перехода в начальную позицию...')
    
    # Переход в начальную позицию
    init_pos = [-0.23, -1, 0.7, 0, -3.143, 0]  # Исправлено с 0.9 на 0.7
    print(f"Переход в начальную позицию: {init_pos}")
    robot.set_realtime_pose(init_pos)
    
    # Ожидание достижения начальной позиции
    if not wait_for_position(robot, init_pos, "Начальная позиция"):
        print("Ошибка: не удалось достичь начальной позиции")
        robot.close()
        exit()

    # Ввод параметров
    # type_obj = int(input('Введите тип предмета (1, 2, 3): '))
    # x = float(input('Введите координату x (м): '))
    # y = float(input('Введите координату y (м): '))


    print("Измеряю координаты COM камерой...")
    xy = None
    for attempt in range(5):
        x, y = measure_xy(display=False, warmup_s=1.5)
        if xy is not None:
            break
        time.sleep(0.2)
    
    if xy is None:
        # fallback to manual
        x = float(input("x (m): "))
        y = float(input("y (m): "))
    else:
        x, y = xy
        print(f"COM по камере: x={x:.4f} м, y={y:.4f} м")

    # Корректировка координат относительно начальной позиции
    z_0 = 0.45
    x_adj = x - 0.23
    y_adj = y - 1
    z_adj = z_0 - 0.3

    # Промежуточная позиция
    intermediate_pos = [-0.23, -1, 0.7, 0, -3.143, 0]
    print(f"Переход в промежуточную позицию: {intermediate_pos}")
    robot.set_realtime_pose(intermediate_pos)
    
    if not wait_for_position(robot, intermediate_pos, "Промежуточная позиция"):
        print("Ошибка: не удалось достичь промежуточной позиции")
        robot.close()
        exit()

    # Пауза в промежуточной позиции
    pause_at_position("Промежуточная позиция")

    # ДОПОЛНИТЕЛЬНАЯ ТОЧКА: Высота z_0 перед спуском к объекту
    approach_pos = [x_adj, y_adj, z_0, 0, -3.143, 0]
    print(f"Переход в точку подхода (высота z_0): {approach_pos}")
    robot.set_realtime_pose(approach_pos)
    
    if not wait_for_position(robot, approach_pos, "Точка подхода (z_0)"):
        print("Ошибка: не удалось достичь точки подхода")
        robot.close()
        exit()

    # Пауза в точке подхода
    pause_at_position("Точка подхода (z_0)")

    # Целевая позиция (нижняя точка)
    target_pos = [x_adj, y_adj, z_adj, 0, -3.143, 0]
    print(f"Переход в целевую позицию (нижняя точка): {target_pos}")
    robot.set_realtime_pose(target_pos)
    
    if not wait_for_position(robot, target_pos, "Нижняя точка"):
        print("Ошибка: не удалось достичь целевой позиции")
        robot.close()
        exit()

    # Пауза в нижней точке
    pause_at_position("Нижняя точка")

    # Возврат в промежуточную позицию
    print("Возврат в промежуточную позицию")
    robot.set_realtime_pose(intermediate_pos)
    wait_for_position(robot, intermediate_pos, "Промежуточная позиция (возврат)")

    # Пауза в промежуточной позиции при возврате
    pause_at_position("Промежуточная позиция (возврат)")

    print("Программа завершена успешно")

except Exception as e:
    print(f"Произошла ошибка: {e}")
    import traceback
    traceback.print_exc()
finally:
    # Всегда закрываем соединение с роботом
    if 'robot' in locals():
        robot.close()