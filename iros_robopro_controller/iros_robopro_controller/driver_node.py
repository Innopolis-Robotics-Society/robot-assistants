from __future__ import annotations

import math
from typing import List

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, TransformStamped
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from std_srvs.srv import SetBool
from tf2_ros import Buffer, TransformBroadcaster, TransformListener, StaticTransformBroadcaster

from iros_custom_msgs.srv import GoToFrame, GripperAction

from .robot_backend import MotionParams, RobotBackend
from .tf_math import Transform, invert, multiply, quat_normalize, rpy_to_quat, robot_pose_deg_to_transform, transform_to_robot_pose_deg


def _get_list_param(node: Node, name: str, n: int, default: List[float]) -> List[float]:
    node.declare_parameter(name, default)
    v = node.get_parameter(name).value
    if not isinstance(v, list) or len(v) != n:
        return default
    return [float(x) for x in v]


class RCRobotDriver(Node):
    def __init__(self) -> None:
        super().__init__('rc_robot_driver')

        # --- Parameters
        self.declare_parameter('robot_ip', '192.168.10.10')
        self.declare_parameter('robot_timeout_sec', 5)

        self.declare_parameter('world_frame', 'world')
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('flange_frame', 'flange_link')
        self.declare_parameter('tool_frame', 'tool0')

        # world -> base_link
        self.world_to_base_xyz = _get_list_param(self, 'world_to_base_xyz', 3, [0.0, 0.0, 0.0])
        self.world_to_base_rpy_deg = _get_list_param(self, 'world_to_base_rpy_deg', 3, [0.0, 0.0, 0.0])

        # flange -> tool0 (TCP offset)
        self.flange_to_tool_xyz = _get_list_param(self, 'flange_to_tool_xyz', 3, [0.0, 0.0, 0.0])
        self.flange_to_tool_rpy_deg = _get_list_param(self, 'flange_to_tool_rpy_deg', 3, [0.0, 0.0, 0.0])
        self.declare_parameter('tool_offset_units', 'deg')
        self.declare_parameter('set_tcp_on_start', True)

        # Motion params
        self.declare_parameter('linear_speed', 0.3)
        self.declare_parameter('linear_accel', 1.0)
        self.declare_parameter('blend', 0.0)
        self.declare_parameter('orientation_units', 'deg')
        self.declare_parameter('move_await_sec', 30)

        # TF lookup
        self.declare_parameter('tf_lookup_timeout_sec', 1.0)

        # Gripper
        self.declare_parameter('gripper_backend', 'wrist')  # wrist|controller
        self.declare_parameter('gripper_pin0', 0)
        self.declare_parameter('gripper_pin1', 1)

        self.robot_ip = str(self.get_parameter('robot_ip').value)
        self.robot_timeout_sec = int(self.get_parameter('robot_timeout_sec').value)

        self.world_frame = str(self.get_parameter('world_frame').value)
        self.base_frame = str(self.get_parameter('base_frame').value)
        self.flange_frame = str(self.get_parameter('flange_frame').value)
        self.tool_frame = str(self.get_parameter('tool_frame').value)

        self.tool_offset_units = str(self.get_parameter('tool_offset_units').value)
        self.set_tcp_on_start = bool(self.get_parameter('set_tcp_on_start').value)

        self.motion_params = MotionParams(
            speed=float(self.get_parameter('linear_speed').value),
            accel=float(self.get_parameter('linear_accel').value),
            blend=float(self.get_parameter('blend').value),
            orientation_units=str(self.get_parameter('orientation_units').value),
            await_sec=int(self.get_parameter('move_await_sec').value),
        )

        self.tf_lookup_timeout = float(self.get_parameter('tf_lookup_timeout_sec').value)

        self.gripper_backend = str(self.get_parameter('gripper_backend').value)
        self.gripper_pin0 = int(self.get_parameter('gripper_pin0').value)
        self.gripper_pin1 = int(self.get_parameter('gripper_pin1').value)

        # --- Robot
        self.robot = RobotBackend(self.robot_ip, timeout=self.robot_timeout_sec, autoconnect=False)
        self._tcp_set = False

        # --- TF
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = TransformBroadcaster(self)
        self.tf_static_broadcaster = StaticTransformBroadcaster(self)

        # Publish static frames immediately
        self._publish_static_frames()

        # --- Publishers
        self.tool_pose_pub = self.create_publisher(PoseStamped, 'tool_pose', 10)
        self.flange_pose_pub = self.create_publisher(PoseStamped, 'flange_pose', 10)

        # --- Services
        cbg = MutuallyExclusiveCallbackGroup()
        self.create_service(GoToFrame, '/go_to_frame', self._srv_go_to_frame, callback_group=cbg)
        self.create_service(GripperAction, '/gripper_action', self._srv_gripper_action, callback_group=cbg)
        self.create_service(SetBool, '/gripper_lock', self._srv_gripper_lock, callback_group=cbg)

        # --- Timers
        self.declare_parameter('publish_rate_hz', 30.0)
        rate_hz = float(self.get_parameter('publish_rate_hz').value)
        period = 1.0 / max(1.0, rate_hz)
        self.create_timer(period, self._timer_publish)

        self.get_logger().info(
            f"Listening services: /go_to_frame, /gripper_action, /gripper_lock | Publishing: tool_pose, flange_pose | TF: {self.world_frame}->{self.base_frame}, {self.base_frame}->{self.flange_frame}->{self.tool_frame}"
        )

    def _publish_static_frames(self) -> None:
        now = self.get_clock().now().to_msg()

        # world -> base
        T_w_b = TransformStamped()
        T_w_b.header.stamp = now
        T_w_b.header.frame_id = self.world_frame
        T_w_b.child_frame_id = self.base_frame
        T_w_b.transform.translation.x = float(self.world_to_base_xyz[0])
        T_w_b.transform.translation.y = float(self.world_to_base_xyz[1])
        T_w_b.transform.translation.z = float(self.world_to_base_xyz[2])
        q_w_b = rpy_to_quat(
            math.radians(self.world_to_base_rpy_deg[0]),
            math.radians(self.world_to_base_rpy_deg[1]),
            math.radians(self.world_to_base_rpy_deg[2]),
        )
        T_w_b.transform.rotation.x = float(q_w_b[0])
        T_w_b.transform.rotation.y = float(q_w_b[1])
        T_w_b.transform.rotation.z = float(q_w_b[2])
        T_w_b.transform.rotation.w = float(q_w_b[3])

        # flange -> tool (TCP)
        T_f_t = TransformStamped()
        T_f_t.header.stamp = now
        T_f_t.header.frame_id = self.flange_frame
        T_f_t.child_frame_id = self.tool_frame
        T_f_t.transform.translation.x = float(self.flange_to_tool_xyz[0])
        T_f_t.transform.translation.y = float(self.flange_to_tool_xyz[1])
        T_f_t.transform.translation.z = float(self.flange_to_tool_xyz[2])
        q_f_t = rpy_to_quat(
            math.radians(self.flange_to_tool_rpy_deg[0]),
            math.radians(self.flange_to_tool_rpy_deg[1]),
            math.radians(self.flange_to_tool_rpy_deg[2]),
        )
        T_f_t.transform.rotation.x = float(q_f_t[0])
        T_f_t.transform.rotation.y = float(q_f_t[1])
        T_f_t.transform.rotation.z = float(q_f_t[2])
        T_f_t.transform.rotation.w = float(q_f_t[3])

        self.tf_static_broadcaster.sendTransform([T_w_b, T_f_t])

    def _ensure_connected(self) -> bool:
        if self.robot.is_connected():
            return True
        ok = self.robot.connect(read_only=False)
        if ok:
            self.get_logger().info('Robot connected')
        return ok

    def _ensure_tcp_set(self) -> bool:
        if self._tcp_set:
            return True

        if not self.set_tcp_on_start:
            self._tcp_set = True
            return True

        tool_end_point = (
            float(self.flange_to_tool_xyz[0]),
            float(self.flange_to_tool_xyz[1]),
            float(self.flange_to_tool_xyz[2]),
            float(self.flange_to_tool_rpy_deg[0]),
            float(self.flange_to_tool_rpy_deg[1]),
            float(self.flange_to_tool_rpy_deg[2]),
        )
        ok = self.robot.set_tcp(tool_end_point=tool_end_point, units=self.tool_offset_units)
        if ok:
            self._tcp_set = True
            self.get_logger().info('TCP set in controller')
        else:
            self.get_logger().warn('Failed to set TCP in controller')
        return ok

    def _timer_publish(self) -> None:
        if not self._ensure_connected():
            return
        self._ensure_tcp_set()

        # Read current TCP pose (base -> tool)
        try:
            tcp = self.robot.get_actual_tcp(orientation_units=self.motion_params.orientation_units)
        except Exception as e:
            self.get_logger().warn(f'get_actual_tcp failed: {e}')
            return

        T_base_tool = robot_pose_deg_to_transform(tcp)

        # Known flange->tool offset
        T_flange_tool = robot_pose_deg_to_transform(
            (
                float(self.flange_to_tool_xyz[0]),
                float(self.flange_to_tool_xyz[1]),
                float(self.flange_to_tool_xyz[2]),
                float(self.flange_to_tool_rpy_deg[0]),
                float(self.flange_to_tool_rpy_deg[1]),
                float(self.flange_to_tool_rpy_deg[2]),
            )
        )
        T_tool_flange = invert(T_flange_tool)
        T_base_flange = multiply(T_base_tool, T_tool_flange)

        stamp = self.get_clock().now().to_msg()

        # Publish PoseStamped: tool
        tool_msg = PoseStamped()
        tool_msg.header.stamp = stamp
        tool_msg.header.frame_id = self.base_frame
        tool_msg.pose.position.x = float(T_base_tool.t[0])
        tool_msg.pose.position.y = float(T_base_tool.t[1])
        tool_msg.pose.position.z = float(T_base_tool.t[2])
        q = quat_normalize(T_base_tool.q)
        tool_msg.pose.orientation.x = float(q[0])
        tool_msg.pose.orientation.y = float(q[1])
        tool_msg.pose.orientation.z = float(q[2])
        tool_msg.pose.orientation.w = float(q[3])
        self.tool_pose_pub.publish(tool_msg)

        # Publish PoseStamped: flange
        flange_msg = PoseStamped()
        flange_msg.header.stamp = stamp
        flange_msg.header.frame_id = self.base_frame
        flange_msg.pose.position.x = float(T_base_flange.t[0])
        flange_msg.pose.position.y = float(T_base_flange.t[1])
        flange_msg.pose.position.z = float(T_base_flange.t[2])
        qf = quat_normalize(T_base_flange.q)
        flange_msg.pose.orientation.x = float(qf[0])
        flange_msg.pose.orientation.y = float(qf[1])
        flange_msg.pose.orientation.z = float(qf[2])
        flange_msg.pose.orientation.w = float(qf[3])
        self.flange_pose_pub.publish(flange_msg)

        # Dynamic TF: base->flange
        tf_bf = TransformStamped()
        tf_bf.header.stamp = stamp
        tf_bf.header.frame_id = self.base_frame
        tf_bf.child_frame_id = self.flange_frame
        tf_bf.transform.translation.x = float(T_base_flange.t[0])
        tf_bf.transform.translation.y = float(T_base_flange.t[1])
        tf_bf.transform.translation.z = float(T_base_flange.t[2])
        tf_bf.transform.rotation.x = float(qf[0])
        tf_bf.transform.rotation.y = float(qf[1])
        tf_bf.transform.rotation.z = float(qf[2])
        tf_bf.transform.rotation.w = float(qf[3])

        self.tf_broadcaster.sendTransform([tf_bf])

    # --- Services

    def _srv_go_to_frame(self, req: GoToFrame.Request, res: GoToFrame.Response) -> GoToFrame.Response:
        target_frame = str(req.frame)

        if not self._ensure_connected():
            res.success = False
            res.message = 'Robot not connected'
            return res
        self._ensure_tcp_set()

        # Look up base -> target_frame (target is defined in world)
        try:
            timeout = Duration(seconds=float(self.tf_lookup_timeout))
            if not self.tf_buffer.can_transform(self.base_frame, target_frame, rclpy.time.Time(), timeout=timeout):
                res.success = False
                res.message = f"No TF from {self.base_frame} to {target_frame}"
                return res

            tf = self.tf_buffer.lookup_transform(self.base_frame, target_frame, rclpy.time.Time(), timeout=timeout)

            T_base_target = Transform(
                t=np.array(
                    [
                        tf.transform.translation.x,
                        tf.transform.translation.y,
                        tf.transform.translation.z,
                    ],
                    dtype=float,
                ),
                q=quat_normalize(
                    np.array(
                        [
                            tf.transform.rotation.x,
                            tf.transform.rotation.y,
                            tf.transform.rotation.z,
                            tf.transform.rotation.w,
                        ],
                        dtype=float,
                    )
                ),
            )

            tcp_pose = transform_to_robot_pose_deg(T_base_target)

        except Exception as e:
            res.success = False
            res.message = f"TF lookup failed: {e}"
            return res

        ok = self.robot.move_linear_tcp(tcp_pose, self.motion_params)
        res.success = bool(ok)
        res.message = 'OK' if ok else 'Move failed'
        return res

    def _srv_gripper_action(self, req: GripperAction.Request, res: GripperAction.Response) -> GripperAction.Response:
        if not self._ensure_connected():
            res.success = False
            res.message = 'Robot not connected'
            return res

        # open=True -> DO0=1,DO1=0 ; open=False -> DO0=0,DO1=1
        do0 = bool(req.open)
        do1 = not bool(req.open)
        ok = self.robot.set_gripper_pins(do0, do1, backend=self.gripper_backend, pin0=self.gripper_pin0, pin1=self.gripper_pin1)

        res.success = bool(ok)
        res.message = 'OK' if ok else 'Failed to set outputs'
        return res

    def _srv_gripper_lock(self, req: SetBool.Request, res: SetBool.Response) -> SetBool.Response:
        if not self._ensure_connected():
            res.success = False
            res.message = 'Robot not connected'
            return res

        # True -> lock (1,1); False -> unlock (0,0)
        do0 = bool(req.data)
        do1 = bool(req.data)
        ok = self.robot.set_gripper_pins(do0, do1, backend=self.gripper_backend, pin0=self.gripper_pin0, pin1=self.gripper_pin1)

        res.success = bool(ok)
        res.message = 'OK' if ok else 'Failed to set outputs'
        return res


def main() -> None:
    rclpy.init()
    node = RCRobotDriver()
    try:
        executor = MultiThreadedExecutor(num_threads=2)
        executor.add_node(node)
        executor.spin()
    finally:
        node.destroy_node()
        rclpy.shutdown()