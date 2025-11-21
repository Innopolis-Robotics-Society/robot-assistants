#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math

import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.duration import Duration  # для tf2 (НЕ путать с msg Duration)

from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from tf2_ros import Buffer, TransformListener, TransformException

from moveit_msgs.srv import GetPositionIK
from moveit_msgs.msg import PositionIKRequest, RobotState
from builtin_interfaces.msg import Duration as DurationMsg  # msg Duration для IK и time_from_start


class GoToTfTarget(Node):
    """
    Node:
      1) читает TF target_frame относительно base_frame
      2) вызывает MoveIt IK (/compute_ik)
      3) публикует JointTrajectory на /scaled_joint_trajectory_controller/joint_trajectory
    """

    def __init__(self):
        super().__init__("go_to_tf_target")

        # ---- параметры ----
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("target_frame", "my_goal")
        self.declare_parameter("group_name", "ur_manipulator")
        self.declare_parameter("ee_link", "tool0")
        self.declare_parameter(
            "traj_topic", "/scaled_joint_trajectory_controller/joint_trajectory"
        )
        self.declare_parameter("ik_service", "/compute_ik")
        self.declare_parameter("move_time", 3.0)

        self.base_frame = (
            self.get_parameter("base_frame").get_parameter_value().string_value
        )
        self.target_frame = (
            self.get_parameter("target_frame").get_parameter_value().string_value
        )
        self.group_name = (
            self.get_parameter("group_name").get_parameter_value().string_value
        )
        self.ee_link = self.get_parameter("ee_link").get_parameter_value().string_value
        self.traj_topic = (
            self.get_parameter("traj_topic").get_parameter_value().string_value
        )
        self.ik_service_name = (
            self.get_parameter("ik_service").get_parameter_value().string_value
        )
        self.move_time = (
            self.get_parameter("move_time").get_parameter_value().double_value
        )

        # ---- TF ----
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # ---- joint_states ----
        self.current_joint_state = None
        self.joint_state_sub = self.create_subscription(
            JointState, "/joint_states", self.joint_state_cb, 10
        )

        # ---- publisher траектории ----
        self.traj_pub = self.create_publisher(
            JointTrajectory, self.traj_topic, 10
        )

        # ---- IK клиент ----
        self.ik_client = self.create_client(GetPositionIK, self.ik_service_name)

        self.get_logger().info(
            f"[GoToTfTarget] base_frame={self.base_frame}, "
            f"target_frame={self.target_frame}, group_name={self.group_name}, "
            f"ee_link={self.ee_link}, traj_topic={self.traj_topic}, "
            f"ik_service={self.ik_service_name}, move_time={self.move_time:.2f}s"
        )

        # Ждем IK сервис (нормально — это делается при старте)
        while not self.ik_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info(
                f"[GoToTfTarget] Waiting for IK service {self.ik_service_name} ..."
            )

        self.get_logger().info(
            f"[GoToTfTarget] IK service {self.ik_service_name} is available."
        )

        # Один таймер, который 1) ждет joint_states и TF, 2) один раз отправляет IK + траекторию
        self.goal_sent = False
        self.timer = self.create_timer(0.5, self.timer_cb)

    # ------------------------------------------------------------------ #
    # callbacks
    # ------------------------------------------------------------------ #
    def joint_state_cb(self, msg: JointState):
        # Просто сохраняем последнее состояние
        self.current_joint_state = msg

    def timer_cb(self):
        # Если уже отправили траекторию — нода ничего больше не делает
        if self.goal_sent:
            return

        # Дождаться нормальных joint_states
        if self.current_joint_state is None or len(self.current_joint_state.name) == 0:
            self.get_logger().warn(
                "[GoToTfTarget] No joint_states received yet, waiting..."
            )
            return

        # Достаем TF base_frame -> target_frame
        try:
            tf = self.tf_buffer.lookup_transform(
                self.base_frame,
                self.target_frame,
                Time(),  # latest
                timeout=Duration(seconds=1.0),
            )
        except TransformException as ex:
            self.get_logger().warn(
                f"[GoToTfTarget] Cannot transform {self.base_frame} -> "
                f"{self.target_frame}: {ex}"
            )
            return

        self.get_logger().info(
            f"[GoToTfTarget] Got TF {self.base_frame} -> {self.target_frame}, calling IK..."
        )

        self._call_ik_async(tf)

    # ------------------------------------------------------------------ #
    # IK + Trajectory
    # ------------------------------------------------------------------ #
    def _call_ik_async(self, tf):
        # Формируем PoseStamped цели
        target_pose = PoseStamped()
        target_pose.header.frame_id = self.base_frame
        target_pose.header.stamp = self.get_clock().now().to_msg()
        target_pose.pose.position.x = tf.transform.translation.x
        target_pose.pose.position.y = tf.transform.translation.y
        target_pose.pose.position.z = tf.transform.translation.z
        target_pose.pose.orientation = tf.transform.rotation

        # RobotState из актуального /joint_states
        robot_state = RobotState()
        robot_state.joint_state = self.current_joint_state

        ik_req = PositionIKRequest()
        ik_req.group_name = self.group_name
        ik_req.ik_link_name = self.ee_link
        ik_req.robot_state = robot_state
        ik_req.avoid_collisions = True
        ik_req.pose_stamped = target_pose
        # ВАЖНО: это msg Duration, а не rclpy.duration.Duration
        ik_req.timeout = DurationMsg(sec=1, nanosec=0)

        req = GetPositionIK.Request()
        req.ik_request = ik_req

        future = self.ik_client.call_async(req)
        # НИКАКИХ spin_until_future_complete внутри callback — только done_callback
        future.add_done_callback(self._ik_response_cb)

    def _ik_response_cb(self, future):
        try:
            res = future.result()
        except Exception as e:
            self.get_logger().error(f"[GoToTfTarget] IK call failed: {e}")
            return

        # MoveIt SUCCESS = 1
        if res.error_code.val != 1:
            self.get_logger().error(
                f"[GoToTfTarget] IK failed with error_code={res.error_code.val}"
            )
            return

        joint_state = res.solution.joint_state
        if not joint_state.name:
            self.get_logger().error(
                "[GoToTfTarget] IK returned empty joint_state, aborting."
            )
            return

        self.get_logger().info(
            f"[GoToTfTarget] IK success, joints: "
            f"{list(zip(joint_state.name, joint_state.position))}"
        )

        # Формируем простую траекторию из одной точки
        traj = JointTrajectory()
        traj.header.stamp = self.get_clock().now().to_msg()
        traj.joint_names = list(joint_state.name)

        point = JointTrajectoryPoint()
        point.positions = list(joint_state.position)
        # msg Duration для time_from_start
        secs = max(0.1, float(self.move_time))
        point.time_from_start = DurationMsg(sec=int(math.ceil(secs)), nanosec=0)

        traj.points.append(point)

        self.traj_pub.publish(traj)
        self.get_logger().info(
            f"[GoToTfTarget] Sent trajectory to {self.traj_topic} "
            f"with move_time={secs:.2f}s"
        )

        self.goal_sent = True
        # Останавливаем таймер, чтобы не спамить IK и траекториями
        if self.timer is not None:
            self.timer.cancel()


def main(args=None):
    rclpy.init(args=args)
    node = GoToTfTarget()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
