import math
import time
import URBasic
import URBasic.robotModel
import URBasic.urScriptExt
from rclpy.node import Node

class RobotUtils:
    """Утилиты для работы с роботом."""
    
    def __init__(self, node: Node):
        self.node = node
        self.robot = None
        self.robot_model = None
        
    @staticmethod
    def normalize_angle(angle):
        """Нормализует угол в диапазон [-pi, pi]."""
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle

    def check_position(self, current_pos, target_pos, linear_tolerance=0.001, angle_tolerance=0.01):
        """Проверяет, достигнута ли целевая позиция."""
        position_reached = True
        
        self.node.get_logger().debug("Проверка позиции:")
        for i, (current, target) in enumerate(zip(current_pos, target_pos)):
            if i < 3:  # Позиция (x, y, z)
                diff = abs(current - target)
                if diff > linear_tolerance:
                    position_reached = False
                status = '✓' if diff <= linear_tolerance else '✗'
                self.node.get_logger().debug(
                    f"Ось {i}(лин.): текущая={current:08.6f}, "
                    f"целевая={target:08.6f}, разница={diff:08.6f} {status}"
                )
            else:  # Ориентация (rx, ry, rz)
                norm_current = self.normalize_angle(current)
                norm_target = self.normalize_angle(target)
                diff = abs(-norm_current - norm_target)
                if diff > angle_tolerance:
                    position_reached = False
                status = '✓' if diff <= angle_tolerance else '✗'
                self.node.get_logger().debug(
                    f"Ось {i}(угл.): текущая={current:08.6f}→{norm_current:08.6f}, "
                    f"целевая={target:08.6f}→{norm_target:08.6f}, "
                    f"разница={diff:08.6f} {status}"
                )
        
        self.node.get_logger().debug(f"Позиция достигнута: {position_reached}")
        return position_reached

    def wait_for_position(self, target_pos, position_name="", timeout=15.0, check_interval=0.05):
        """Ожидает достижения целевой позиции с таймаутом."""
        start_time = time.time()
        attempt = 0
        
        while time.time() - start_time < timeout:
            attempt += 1
            current_pos = self.robot.get_actual_tcp_pose_custom()
            
            if self.check_position(current_pos, target_pos):
                self.node.get_logger().info(f"✓ Позиция '{position_name}' успешно достигнута!")
                return True
            
            if attempt % 10 == 0:
                elapsed = time.time() - start_time
                self.node.get_logger().info(f"Прогресс: {elapsed:.1f}с из {timeout}с")
            
            time.sleep(check_interval)
        
        self.node.get_logger().error(f"✗ Таймаут: позиция '{position_name}' не достигнута за {timeout} секунд")
        return False

    def initialize_robot(self, host):
        """Инициализация соединения с роботом."""
        self.node.get_logger().info("Инициализация UR робота")
        self.robot_model = URBasic.robotModel.RobotModel()
        self.robot = URBasic.urScriptExt.UrScriptExt(
            host=host, 
            robotModel=self.robot_model
        )
        self.robot.init_realtime_control()
        
        current_pos = self.robot.get_actual_tcp_pose_custom()
        self.node.get_logger().info(
            f'Текущая позиция робота: '
            f'[{current_pos[0]:08.6f}, {current_pos[1]:08.6f}, {current_pos[2]:08.6f}, '
            f'{current_pos[3]:08.6f}, {current_pos[4]:08.6f}, {current_pos[5]:08.6f}]'
        )
        
        return self.robot

    def move_to_position(self, target_pos, position_name):
        """Перемещение робота в указанную позицию."""
        self.node.get_logger().info(f"Переход в позицию '{position_name}': {target_pos}")
        self.robot.set_realtime_pose(target_pos)
        
        if not self.wait_for_position(target_pos, position_name):
            self.node.get_logger().error(f"Ошибка: не удалось достичь позиции '{position_name}'")
            return False
        return True

    @staticmethod
    def pause_at_position(position_name, duration=2.0):
        """Пауза в достигнутой позиции."""
        print(f"⏸️  Пауза {duration} секунд в позиции '{position_name}'...")
        time.sleep(duration)
        print(f"▶️  Пауза завершена, продолжаем работу.")
