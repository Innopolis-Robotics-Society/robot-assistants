#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose
from apriltag_msgs.msg import AprilTagDetectionArray

class AprilTagToNav2(Node):
    def __init__(self):
        super().__init__('apriltag_to_nav2')
        
        # Параметры (можно изменить через launch-файл)
        self.declare_parameter('camera_frame', 'camera')
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('goal_topic', '/goal_pose')
        
        self.camera_frame = self.get_parameter('camera_frame').value
        self.base_frame = self.get_parameter('base_frame').value
        self.goal_topic = self.get_parameter('goal_topic').value
        
        # TF2
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # Action Client для Nav2
        self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        
        # Подписка на AprilTag детекцию
        self.tag_sub = self.create_subscription(
            AprilTagDetectionArray,
            '/detections',
            self.tag_callback,
            10
        )
        
        self.get_logger().info(f"AprilTagToNav2 инициализирован. Ожидание AprilTag детекций...")

    def tag_callback(self, msg):
        if not msg.detections:
            return
            
        for detection in msg.detections:
            tag_id = detection.id
            tag_frame = f"marker{tag_id}"
            
            try:
                # Получаем преобразование от камеры к маркеру
                transform = self.tf_buffer.lookup_transform(
                    self.camera_frame,
                    tag_frame,
                    rclpy.time.Time()
                )
                
                # Создаем PoseStamped в фрейме камеры
                camera_pose = PoseStamped()
                camera_pose.header = transform.header
                camera_pose.pose.position.x = transform.transform.translation.x
                camera_pose.pose.position.y = transform.transform.translation.y
                camera_pose.pose.position.z = transform.transform.translation.z
                camera_pose.pose.orientation = transform.transform.rotation
                
                # Преобразуем в базовый фрейм (например, base_link или map)
                try:
                    base_transform = self.tf_buffer.lookup_transform(
                        self.base_frame,
                        self.camera_frame,
                        rclpy.time.Time()
                    )
                    
                    # Применяем преобразование
                    goal_pose = self.do_transform_pose(camera_pose, base_transform)
                    goal_pose.header.stamp = self.get_clock().now().to_msg()
                    goal_pose.header.frame_id = self.base_frame
                    
                    self.get_logger().info(f"Найдена цель по AprilTag {tag_id}. Отправка в Nav2...")
                    self.send_nav2_goal(goal_pose)
                    
                except TransformException as ex:
                    self.get_logger().error(f"Ошибка преобразования в базовый фрейм: {ex}")
                
            except TransformException as ex:
                self.get_logger().error(f"Ошибка преобразования для AprilTag {tag_id}: {ex}")

    def do_transform_pose(self, pose, transform):
        """ Преобразует PoseStamped с помощью transform """
        new_pose = PoseStamped()
        new_pose.header = transform.header
        new_pose.pose.position.x = (
            transform.transform.translation.x +
            pose.pose.position.x
        )
        new_pose.pose.position.y = (
            transform.transform.translation.y +
            pose.pose.position.y
        )
        new_pose.pose.position.z = (
            transform.transform.translation.z +
            pose.pose.position.z
        )
        new_pose.pose.orientation = transform.transform.rotation
        return new_pose

    def send_nav2_goal(self, pose):
        """ Отправляет цель в Nav2 через Action """
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = pose
        
        if not self.nav_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("Nav2 Action Server недоступен!")
            return
            
        self.get_logger().info("Отправка цели в Nav2...")
        self._send_goal_future = self.nav_client.send_goal_async(
            goal_msg,
            feedback_callback=self.feedback_callback
        )
        self._send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error("Цель отклонена Nav2")
            return
            
        self.get_logger().info("Цель принята Nav2!")
        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.result_callback)

    def feedback_callback(self, feedback_msg):
        feedback = feedback_msg.feedback
        self.get_logger().info(f"Осталось до цели: {feedback.distance_remaining:.2f} м")

    def result_callback(self, future):
        result = future.result().result
        self.get_logger().info(f"Результат навигации: {result}")

def main(args=None):
    rclpy.init(args=args)
    node = AprilTagToNav2()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()