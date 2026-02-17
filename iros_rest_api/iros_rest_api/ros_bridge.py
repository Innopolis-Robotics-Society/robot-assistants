from __future__ import annotations

import threading
import time
import traceback
import uuid
from typing import Any, Callable, Dict, List, Optional, Tuple

import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import Bool, String
from std_srvs.srv import Trigger

from iros_custom_msgs.srv import DetectObject, GoToFrame, GripperAction

from .schemas import CommandRecord, ServiceCallResult


DEFAULT_SERVICE_NAMES = {
    "detect_object": "/detect_object",
    "go_to_frame": "/go_to_frame",
    "gripper_action": "/gripper_action",
    "cv_run_rust": "/iros_cv_session_hub/run_rust",
    "cv_run_pcb": "/iros_cv_session_hub/run_pcb",
    "cv_run_gear": "/iros_cv_session_hub/run_gear",
    "cv_publish": "/iros_cv_session_hub/publish",
}


class RosApiBridge:
    """Small ROS2 bridge used by the FastAPI process.

    The FastAPI server runs in the main thread, while rclpy spins in a background
    executor thread. REST handlers call ROS services through this object and can
    also submit longer calls as async commands with command_id tracking.
    """

    def __init__(self, node_name: str = "iros_rest_api") -> None:
        self.node_name = node_name
        self.node: Optional[Node] = None
        self.executor: Optional[MultiThreadedExecutor] = None
        self._spin_thread: Optional[threading.Thread] = None
        self._started = False
        self._lock = threading.RLock()
        self._last_error: Optional[str] = None
        self._started_at = time.time()

        self._commands: Dict[str, CommandRecord] = {}
        self._commands_lock = threading.RLock()

        self._snapshot_lock = threading.RLock()
        self._cv_ok: Optional[bool] = None
        self._cv_report: Optional[str] = None
        self._voice_executor_status: Optional[str] = None
        self._snapshot_updated_at: Dict[str, float] = {}

        self._voice_publishers: Dict[str, Any] = {}

        self.service_names = dict(DEFAULT_SERVICE_NAMES)
        self.service_types = {
            "detect_object": DetectObject,
            "go_to_frame": GoToFrame,
            "gripper_action": GripperAction,
            "cv_run_rust": Trigger,
            "cv_run_pcb": Trigger,
            "cv_run_gear": Trigger,
            "cv_publish": Trigger,
        }

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def start(self) -> None:
        with self._lock:
            if self._started:
                return
            if not rclpy.ok():
                rclpy.init(args=None)

            self.node = Node(self.node_name)
            self.executor = MultiThreadedExecutor(num_threads=4)
            self.executor.add_node(self.node)

            self._create_subscriptions()

            self._spin_thread = threading.Thread(target=self._spin, name="ros2-api-spin", daemon=True)
            self._spin_thread.start()
            self._started = True
            self.node.get_logger().info("ROS REST bridge started")

    def stop(self) -> None:
        with self._lock:
            if not self._started:
                return
            try:
                if self.executor is not None:
                    self.executor.shutdown()
                if self.node is not None:
                    self.node.destroy_node()
            finally:
                self._started = False
                self.node = None
                self.executor = None

    def _spin(self) -> None:
        assert self.executor is not None
        try:
            self.executor.spin()
        except Exception as exc:  # pragma: no cover - defensive runtime guard
            self._last_error = f"ROS executor crashed: {exc}\n{traceback.format_exc()}"

    def _require_node(self) -> Node:
        if self.node is None:
            raise RuntimeError("ROS bridge is not started")
        return self.node

    # ------------------------------------------------------------------
    # Subscriptions / state
    # ------------------------------------------------------------------
    def _create_subscriptions(self) -> None:
        node = self._require_node()
        node.create_subscription(Bool, "/cv_hub/ok", self._on_cv_ok, 10)
        node.create_subscription(String, "/cv_hub/report", self._on_cv_report, 10)
        node.create_subscription(String, "voice/executor_status", self._on_voice_status, 10)

    def _on_cv_ok(self, msg: Bool) -> None:
        with self._snapshot_lock:
            self._cv_ok = bool(msg.data)
            self._snapshot_updated_at["cv_ok"] = time.time()

    def _on_cv_report(self, msg: String) -> None:
        with self._snapshot_lock:
            self._cv_report = str(msg.data)
            self._snapshot_updated_at["cv_report"] = time.time()

    def _on_voice_status(self, msg: String) -> None:
        with self._snapshot_lock:
            self._voice_executor_status = str(msg.data)
            self._snapshot_updated_at["voice_executor_status"] = time.time()

    def snapshot(self) -> Dict[str, Any]:
        with self._snapshot_lock:
            return {
                "cv_ok": self._cv_ok,
                "cv_report": self._cv_report,
                "voice_executor_status": self._voice_executor_status,
                "updated_at": dict(self._snapshot_updated_at),
            }

    # ------------------------------------------------------------------
    # Health / discovery
    # ------------------------------------------------------------------
    @property
    def uptime_s(self) -> float:
        return time.time() - self._started_at

    @property
    def last_error(self) -> Optional[str]:
        return self._last_error

    def list_services(self) -> List[Tuple[str, List[str]]]:
        node = self._require_node()
        return [(name, list(types)) for name, types in node.get_service_names_and_types()]

    def known_service_availability(self) -> Dict[str, bool]:
        return {
            alias: self.service_available(
                service_name=service_name,
                srv_type=self.service_types.get(alias, Trigger),
                timeout_s=0.05,
            )
            for alias, service_name in self.service_names.items()
        }

    def service_available(self, service_name: str, srv_type: Any = Trigger, timeout_s: float = 0.1) -> bool:
        node = self._require_node()
        client = node.create_client(srv_type, service_name)
        try:
            return bool(client.wait_for_service(timeout_sec=timeout_s))
        finally:
            node.destroy_client(client)

    # ------------------------------------------------------------------
    # Sync ROS service calls
    # ------------------------------------------------------------------
    def detect_object(self, class_name: str, duration: float, timeout_s: float = 10.0) -> ServiceCallResult:
        req = DetectObject.Request()
        req.class_name = class_name
        req.duration = float(duration)
        return self._call_service(
            srv_type=DetectObject,
            service_name=self.service_names["detect_object"],
            request=req,
            timeout_s=timeout_s,
            success_field="accepted",
        )

    def go_to_frame(self, frame: str, timeout_s: float = 30.0) -> ServiceCallResult:
        req = GoToFrame.Request()
        req.frame = frame
        return self._call_service(
            srv_type=GoToFrame,
            service_name=self.service_names["go_to_frame"],
            request=req,
            timeout_s=timeout_s,
            success_field="success",
        )

    def gripper(self, open_: bool, timeout_s: float = 10.0) -> ServiceCallResult:
        req = GripperAction.Request()
        req.open = bool(open_)
        return self._call_service(
            srv_type=GripperAction,
            service_name=self.service_names["gripper_action"],
            request=req,
            timeout_s=timeout_s,
            success_field="success",
        )

    def run_cv(self, target: str, timeout_s: float = 20.0) -> ServiceCallResult:
        target_to_alias = {
            "rust": "cv_run_rust",
            "pcb": "cv_run_pcb",
            "gear": "cv_run_gear",
            "publish": "cv_publish",
        }
        if target not in target_to_alias:
            raise ValueError(f"Unsupported CV target: {target}")
        return self._trigger(self.service_names[target_to_alias[target]], timeout_s=timeout_s)

    def _trigger(self, service_name: str, timeout_s: float = 10.0) -> ServiceCallResult:
        req = Trigger.Request()
        return self._call_service(
            srv_type=Trigger,
            service_name=service_name,
            request=req,
            timeout_s=timeout_s,
            success_field="success",
        )

    def _call_service(
        self,
        *,
        srv_type: Any,
        service_name: str,
        request: Any,
        timeout_s: float,
        success_field: str,
    ) -> ServiceCallResult:
        node = self._require_node()
        start = time.time()
        client = node.create_client(srv_type, service_name)
        try:
            deadline = time.time() + timeout_s
            service_ready = False
            while time.time() < deadline:
                if client.wait_for_service(timeout_sec=0.1):
                    service_ready = True
                    break
            if not service_ready:
                latency_ms = (time.time() - start) * 1000.0
                return ServiceCallResult(
                    success=False,
                    message=f"Service is not available: {service_name}",
                    service_name=service_name,
                    latency_ms=latency_ms,
                    raw={},
                )

            future = client.call_async(request)
            while time.time() < deadline:
                if future.done():
                    response = future.result()
                    latency_ms = (time.time() - start) * 1000.0
                    success = bool(getattr(response, success_field, False))
                    message = str(getattr(response, "message", ""))
                    return ServiceCallResult(
                        success=success,
                        message=message,
                        service_name=service_name,
                        latency_ms=latency_ms,
                        raw=self._ros_response_to_dict(response),
                    )
                time.sleep(0.01)

            latency_ms = (time.time() - start) * 1000.0
            return ServiceCallResult(
                success=False,
                message=f"Timeout after {timeout_s:.1f}s while waiting for {service_name}",
                service_name=service_name,
                latency_ms=latency_ms,
                raw={},
            )
        except Exception as exc:
            self._last_error = f"Service call failed: {service_name}: {exc}"
            latency_ms = (time.time() - start) * 1000.0
            return ServiceCallResult(
                success=False,
                message=str(exc),
                service_name=service_name,
                latency_ms=latency_ms,
                raw={},
            )
        finally:
            node.destroy_client(client)

    @staticmethod
    def _ros_response_to_dict(response: Any) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        slots = getattr(response, "__slots__", [])
        for slot in slots:
            name = slot[1:] if slot.startswith("_") else slot
            try:
                value = getattr(response, name)
            except AttributeError:
                continue
            result[name] = value
        return result

    # ------------------------------------------------------------------
    # Publishers
    # ------------------------------------------------------------------
    def publish_voice_command(self, command: str, topic: str = "voice/command") -> Dict[str, Any]:
        node = self._require_node()
        if topic not in self._voice_publishers:
            self._voice_publishers[topic] = node.create_publisher(String, topic, 10)
        msg = String()
        msg.data = command
        self._voice_publishers[topic].publish(msg)
        return {"published": True, "topic": topic, "command": command}

    # ------------------------------------------------------------------
    # Async command tracking
    # ------------------------------------------------------------------
    def submit_command(
        self,
        *,
        kind: str,
        request: Dict[str, Any],
        fn: Callable[[], Any],
    ) -> CommandRecord:
        command_id = str(uuid.uuid4())
        now = time.time()
        record = CommandRecord(
            command_id=command_id,
            kind=kind,
            status="pending",
            created_at=now,
            updated_at=now,
            request=request,
        )
        with self._commands_lock:
            self._commands[command_id] = record

        thread = threading.Thread(
            target=self._run_command_thread,
            args=(command_id, fn),
            name=f"rest-command-{command_id[:8]}",
            daemon=True,
        )
        thread.start()
        return record

    def _run_command_thread(self, command_id: str, fn: Callable[[], Any]) -> None:
        self._update_command(command_id, status="running")
        try:
            result = fn()
            if hasattr(result, "dict"):
                payload = result.dict()
            else:
                payload = dict(result) if isinstance(result, dict) else {"value": result}
            success = bool(payload.get("success", True))
            self._update_command(command_id, status="success" if success else "failed", result=payload)
        except Exception as exc:  # pragma: no cover - defensive runtime guard
            self._update_command(command_id, status="failed", error=str(exc))

    def _update_command(
        self,
        command_id: str,
        *,
        status: Optional[str] = None,
        result: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
    ) -> None:
        with self._commands_lock:
            record = self._commands[command_id]
            data = record.dict()
            if status is not None:
                data["status"] = status
            if result is not None:
                data["result"] = result
            if error is not None:
                data["error"] = error
            data["updated_at"] = time.time()
            self._commands[command_id] = CommandRecord(**data)

    def get_command(self, command_id: str) -> Optional[CommandRecord]:
        with self._commands_lock:
            return self._commands.get(command_id)

    def list_commands(self) -> List[CommandRecord]:
        with self._commands_lock:
            return sorted(self._commands.values(), key=lambda x: x.created_at, reverse=True)
