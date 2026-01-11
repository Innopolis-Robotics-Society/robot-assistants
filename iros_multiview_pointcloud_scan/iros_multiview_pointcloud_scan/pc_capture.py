#!/usr/bin/env python3
# ROS 2 Humble (rclpy)
#
# Orchestrator node:
# 1) (optional) /cloud_accumulator/reset
# 2) for frame in [view_0..view_5]:
#       call /go_to_frame (GoToFrame) -> wait success
#       sleep settle_time
#       call /cloud_accumulator/capture (Trigger)  [recommended for multi-view merge]
#       (optional) call /cloud_accumulator/save after each view (with save_path template)
# 3) call /cloud_accumulator/save once at the end
#
# Start:
#   ros2 run <your_pkg> scan_sequence_runner
#   ros2 service call /scan_sequence_runner/run std_srvs/srv/Trigger {}

from __future__ import annotations

import os
import time
import threading
from typing import List, Optional, Tuple

import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger

from iros_assistant_bringup.srv import GoToFrame

# Optional: set /cloud_accumulator parameter save_path to save per-view files
from rcl_interfaces.srv import SetParameters
from rcl_interfaces.msg import Parameter, ParameterValue


def wait_future(fut, timeout_sec: float) -> Tuple[bool, Optional[object]]:
    """Wait until future done. Main executor must be spinning elsewhere."""
    t0 = time.time()
    while time.time() - t0 < timeout_sec:
        if fut.done():
            return True, fut.result()
        time.sleep(0.01)
    return False, None


class ScanSequenceRunner(Node):
    def __init__(self) -> None:
        super().__init__("scan_sequence_runner")

        # Parameters
        self.declare_parameter("frames", ["view_0", "view_1", "view_2", "view_3", "view_4", "view_5"])

        self.declare_parameter("go_to_frame_service", "/go_to_frame")
        self.declare_parameter("go_timeout_sec", 120.0)

        # Cloud accumulator services (from your accumulator node)
        self.declare_parameter("acc_reset_service", "/pointcloud_accumulator/reset")
        self.declare_parameter("acc_capture_service", "/pointcloud_accumulator/capture")
        self.declare_parameter("acc_save_service", "/pointcloud_accumulator/save")
        self.declare_parameter("acc_publish_service", "/pointcloud_accumulator/publish")  # optional

        self.declare_parameter("do_reset", True)
        self.declare_parameter("do_capture_each_view", True)
        self.declare_parameter("do_publish_end", False)

        self.declare_parameter("settle_time_sec", 0.35)
        self.declare_parameter("trigger_timeout_sec", 10.0)

        # Optional: save after each view by updating accumulator param save_path
        self.declare_parameter("save_each_view", False)
        self.declare_parameter("acc_node_name", "/pointcloud_accumulator")  # node name for set_parameters
        self.declare_parameter("save_path_template", "/tmp/scan_{frame}.ply")
        self.declare_parameter("save_path_end", "/tmp/scan_merged.ply")

        self.frames: List[str] = list(self.get_parameter("frames").value)
        self.go_srv = str(self.get_parameter("go_to_frame_service").value)
        self.go_timeout = float(self.get_parameter("go_timeout_sec").value)

        self.acc_reset_srv = str(self.get_parameter("acc_reset_service").value)
        self.acc_capture_srv = str(self.get_parameter("acc_capture_service").value)
        self.acc_save_srv = str(self.get_parameter("acc_save_service").value)
        self.acc_publish_srv = str(self.get_parameter("acc_publish_service").value)

        self.do_reset = bool(self.get_parameter("do_reset").value)
        self.do_capture_each_view = bool(self.get_parameter("do_capture_each_view").value)
        self.do_publish_end = bool(self.get_parameter("do_publish_end").value)

        self.settle_time = float(self.get_parameter("settle_time_sec").value)
        self.trig_timeout = float(self.get_parameter("trigger_timeout_sec").value)

        self.save_each_view = bool(self.get_parameter("save_each_view").value)
        self.acc_node_name = str(self.get_parameter("acc_node_name").value)
        self.save_path_template = str(self.get_parameter("save_path_template").value)
        self.save_path_end = str(self.get_parameter("save_path_end").value)

        # Clients
        self.go_cli = self.create_client(GoToFrame, self.go_srv)

        self.reset_cli = self.create_client(Trigger, self.acc_reset_srv)
        self.capture_cli = self.create_client(Trigger, self.acc_capture_srv)
        self.save_cli = self.create_client(Trigger, self.acc_save_srv)
        self.publish_cli = self.create_client(Trigger, self.acc_publish_srv)

        # Parameter set client for accumulator (optional)
        self.set_params_cli = self.create_client(SetParameters, f"{self.acc_node_name}/set_parameters")

        # Control services
        self.srv_run = self.create_service(Trigger, "~/run", self._run_cb)
        self.srv_stop = self.create_service(Trigger, "~/stop", self._stop_cb)

        self._lock = threading.Lock()
        self._running = False
        self._stop_requested = False
        self._worker: Optional[threading.Thread] = None

        self.get_logger().info(
            f"Frames={self.frames}\n"
            f"go_to_frame={self.go_srv}\n"
            f"acc: reset={self.acc_reset_srv} capture={self.acc_capture_srv} save={self.acc_save_srv}\n"
            f"do_reset={self.do_reset} do_capture_each_view={self.do_capture_each_view}\n"
            f"save_each_view={self.save_each_view} save_end={self.save_path_end}"
        )

    # ------------------ control ------------------

    def _run_cb(self, req, resp: Trigger.Response) -> Trigger.Response:
        with self._lock:
            if self._running:
                resp.success = False
                resp.message = "Already running."
                return resp
            self._running = True
            self._stop_requested = False

        # Quick availability checks
        if not self.go_cli.wait_for_service(timeout_sec=2.0):
            self._set_running(False)
            resp.success = False
            resp.message = f"Service not available: {self.go_srv}"
            return resp

        for cli, name in [
            (self.save_cli, self.acc_save_srv),
            (self.capture_cli, self.acc_capture_srv),
            (self.reset_cli, self.acc_reset_srv),
        ]:
            if not cli.wait_for_service(timeout_sec=2.0):
                self._set_running(False)
                resp.success = False
                resp.message = f"Service not available: {name}"
                return resp

        if self.save_each_view and not self.set_params_cli.wait_for_service(timeout_sec=2.0):
            self.get_logger().warn(
                f"save_each_view=true but {self.acc_node_name}/set_parameters not available; "
                f"will still run but per-view save will fail."
            )

        self._worker = threading.Thread(target=self._run_sequence, daemon=True)
        self._worker.start()

        resp.success = True
        resp.message = "Started."
        return resp

    def _stop_cb(self, req, resp: Trigger.Response) -> Trigger.Response:
        with self._lock:
            self._stop_requested = True
        resp.success = True
        resp.message = "Stop requested."
        return resp

    def _set_running(self, val: bool) -> None:
        with self._lock:
            self._running = val

    def _is_stop(self) -> bool:
        with self._lock:
            return self._stop_requested

    # ------------------ service calls ------------------

    def _call_trigger(self, cli, timeout: float) -> Tuple[bool, str]:
        fut = cli.call_async(Trigger.Request())
        ok, res = wait_future(fut, timeout)
        if not ok or res is None:
            return False, "timeout"
        return bool(res.success), str(res.message)

    def _call_go_to_frame(self, frame: str) -> Tuple[bool, str]:
        req = GoToFrame.Request()
        req.frame = frame
        fut = self.go_cli.call_async(req)
        ok, res = wait_future(fut, self.go_timeout)
        if not ok or res is None:
            return False, "timeout"
        return bool(res.success), str(res.message)

    def _set_acc_save_path(self, path: str) -> Tuple[bool, str]:
        if not self.set_params_cli.service_is_ready():
            return False, "set_parameters service not ready"

        req = SetParameters.Request()
        p = Parameter()
        p.name = "save_path"
        pv = ParameterValue()
        pv.type = ParameterValue.TYPE_STRING
        pv.string_value = path
        p.value = pv
        req.parameters = [p]

        fut = self.set_params_cli.call_async(req)
        ok, res = wait_future(fut, self.trig_timeout)
        if not ok or res is None or len(res.results) == 0:
            return False, "timeout/no result"
        if not res.results[0].successful:
            return False, res.results[0].reason
        return True, "OK"

    # ------------------ main sequence ------------------

    def _run_sequence(self) -> None:
        try:
            # Optional reset
            if self.do_reset:
                ok, msg = self._call_trigger(self.reset_cli, self.trig_timeout)
                if not ok:
                    self.get_logger().error(f"reset failed: {msg}")
                    return
                self.get_logger().info(f"reset: {msg}")

            for frame in self.frames:
                if self._is_stop():
                    self.get_logger().warn("Stopped by request.")
                    return

                # Move
                ok, msg = self._call_go_to_frame(frame)
                if not ok:
                    self.get_logger().error(f"go_to_frame({frame}) failed: {msg}")
                    return
                self.get_logger().info(f"arrived {frame}: {msg}")

                # settle
                time.sleep(self.settle_time)

                # Capture cloud at this view (recommended for multi-view merge)
                if self.do_capture_each_view:
                    ok, msg = self._call_trigger(self.capture_cli, self.trig_timeout)
                    if not ok:
                        self.get_logger().error(f"capture failed at {frame}: {msg}")
                        return
                    self.get_logger().info(f"capture {frame}: {msg}")

                # Optional save after each view (requires accumulator param save_path)
                if self.save_each_view:
                    path = self.save_path_template.format(frame=frame)
                    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

                    ok, msg = self._set_acc_save_path(path)
                    if not ok:
                        self.get_logger().error(f"set save_path failed at {frame}: {msg}")
                        return

                    ok, msg = self._call_trigger(self.save_cli, self.trig_timeout)
                    if not ok:
                        self.get_logger().error(f"save failed at {frame}: {msg}")
                        return
                    self.get_logger().info(f"saved {frame} -> {path}: {msg}")

            # End: save merged cloud
            # Set save_path_end (optional; depends on your accumulator supporting save_path param)
            if self.save_path_end:
                os.makedirs(os.path.dirname(self.save_path_end) or ".", exist_ok=True)
                ok, msg = self._set_acc_save_path(self.save_path_end)
                if not ok:
                    self.get_logger().warn(f"Could not set save_path_end ({self.save_path_end}): {msg}")

            ok, msg = self._call_trigger(self.save_cli, self.trig_timeout)
            if not ok:
                self.get_logger().error(f"final save failed: {msg}")
                return
            self.get_logger().info(f"final save: {msg} (path={self.save_path_end})")

            if self.do_publish_end and self.publish_cli.service_is_ready():
                ok, msg = self._call_trigger(self.publish_cli, self.trig_timeout)
                if ok:
                    self.get_logger().info(f"publish: {msg}")
                else:
                    self.get_logger().warn(f"publish failed: {msg}")

        finally:
            self._set_running(False)
            self.get_logger().info("Sequence finished.")

def main():
    rclpy.init()
    node = ScanSequenceRunner()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
