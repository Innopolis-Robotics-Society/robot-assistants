#!/usr/bin/env python3
import json
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from std_msgs.msg import String

from rosidl_runtime_py.utilities import get_service, get_message
from rosidl_runtime_py.set_message import set_message_fields

try:
    import yaml
except ImportError:
    yaml = None


# ---------- Step model ----------

@dataclass
class StepSpec:
    step_type: str

    # service step
    service_name: str = ""
    service_type: str = ""
    request: Dict[str, Any] = field(default_factory=dict)
    success_field: str = "success"
    message_field: str = "message"

    # wait_topic step
    topic_name: str = ""
    topic_type: str = ""          # e.g. "std_msgs/msg/Bool"
    store_as: str = ""            # e.g. "cv_ok"
    timeout_sec: float = 0.0

    # branch step
    var: str = ""                 # e.g. "cv_ok"
    cases: Dict[Any, List[Dict[str, Any]]] = field(default_factory=dict)  # raw dict steps

    # sleep step
    duration_sec: float = 0.0


def _parse_bool_like(v: Any) -> Optional[bool]:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        if v == 0:
            return False
        if v == 1:
            return True
        return None
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("true", "1", "yes", "y", "ok"):
            return True
        if s in ("false", "0", "no", "n", "nok"):
            return False
    return None


class ServiceClientCache:
    def __init__(self, node: Node):
        self._node = node
        self._clients: Dict[Tuple[str, str], Any] = {}
        self._cb_group = MutuallyExclusiveCallbackGroup()

    def get_client(self, srv_type_str: str, srv_name: str):
        key = (srv_type_str, srv_name)
        if key in self._clients:
            return self._clients[key]

        srv_cls = get_service(srv_type_str)
        client = self._node.create_client(srv_cls, srv_name, callback_group=self._cb_group)
        self._clients[key] = client
        return client


class RoutineRunner:
    def __init__(self, node: Node, client_cache: ServiceClientCache, steps: List[StepSpec],
                 context: Dict[str, Any], done_cb):
        self._node = node
        self._client_cache = client_cache
        self._steps = steps
        self._context = context
        self._done_cb = done_cb
        self._idx = 0

        self._active_sub = None
        self._timeout_timer = None
        self._sleep_timer = None

    def start(self):
        self._run_next()

    def _run_next(self):
        if self._idx >= len(self._steps):
            self._done_cb(True, "routine completed")
            return

        step = self._steps[self._idx]

        if step.step_type == "service":
            self._run_service(step); return
        if step.step_type == "wait_topic":
            self._run_wait_topic(step); return
        if step.step_type == "branch":
            self._run_branch(step); return
        if step.step_type == "sleep":
            self._run_sleep(step); return

        self._done_cb(False, f"unsupported step_type: {step.step_type}")

    # ---- service ----
    def _run_service(self, step: StepSpec):
        client = self._client_cache.get_client(step.service_type, step.service_name)

        if not client.service_is_ready():
            self._node.get_logger().info(
                f"[step {self._idx}] waiting for service {step.service_name} ({step.service_type})..."
            )
            if not client.wait_for_service(timeout_sec=2.0):
                self._done_cb(False, f"service not available: {step.service_name}")
                return

        srv_cls = get_service(step.service_type)
        req = srv_cls.Request()

        # "$var" templating from context
        req_dict = {}
        for k, v in (step.request or {}).items():
            if isinstance(v, str) and v.startswith("$"):
                req_dict[k] = self._context.get(v[1:], v)
            else:
                req_dict[k] = v

        set_message_fields(req, req_dict)

        self._node.get_logger().info(
            f"[step {self._idx}] call {step.service_name} {step.service_type} req={req_dict}"
        )
        future = client.call_async(req)
        future.add_done_callback(lambda fut: self._on_service_done(step, fut))

    def _on_service_done(self, step: StepSpec, future):
        try:
            resp = future.result()
        except Exception as e:
            self._done_cb(False, f"service call failed: {step.service_name}: {e}")
            return

        ok = True
        msg = ""
        if hasattr(resp, step.success_field):
            ok = bool(getattr(resp, step.success_field))
        if hasattr(resp, step.message_field):
            msg = str(getattr(resp, step.message_field))

        if not ok:
            self._done_cb(False, f"step failed: {step.service_name}: {msg}".strip())
            return

        self._node.get_logger().info(f"[step {self._idx}] ok: {step.service_name} {msg}".strip())
        self._idx += 1
        self._run_next()

    # ---- wait topic ----
    def _run_wait_topic(self, step: StepSpec):
        if not step.topic_name or not step.topic_type or not step.store_as:
            self._done_cb(False, "wait_topic: topic_name/topic_type/store_as required")
            return

        msg_cls = get_message(step.topic_type)
        self._cleanup_wait_resources()

        self._node.get_logger().info(
            f"[step {self._idx}] wait topic {step.topic_name} ({step.topic_type}) -> store '{step.store_as}'"
        )

        def _cb(msg):
            val = getattr(msg, "data", None)

            if step.topic_type in ("std_msgs/msg/Bool",) or step.topic_type.endswith("/Bool"):
                parsed = bool(val)
            else:
                b = _parse_bool_like(val)
                parsed = b if b is not None else val

            self._context[step.store_as] = parsed
            self._node.get_logger().info(f"[step {self._idx}] got {step.topic_name}: {parsed!r}")

            self._cleanup_wait_resources()
            self._idx += 1
            self._run_next()

        self._active_sub = self._node.create_subscription(msg_cls, step.topic_name, _cb, 10)

        if step.timeout_sec and step.timeout_sec > 0:
            self._timeout_timer = self._node.create_timer(step.timeout_sec, lambda: self._on_wait_timeout(step))

    def _on_wait_timeout(self, step: StepSpec):
        self._cleanup_wait_resources()
        self._done_cb(False, f"timeout waiting for topic: {step.topic_name}")

    def _cleanup_wait_resources(self):
        if self._timeout_timer is not None:
            try:
                self._node.destroy_timer(self._timeout_timer)
            except Exception:
                pass
            self._timeout_timer = None

        if self._active_sub is not None:
            try:
                self._node.destroy_subscription(self._active_sub)
            except Exception:
                pass
            self._active_sub = None

    # ---- branch ----
    def _run_branch(self, step: StepSpec):
        if not step.var or not step.cases:
            self._done_cb(False, "branch: var/cases required")
            return

        val = self._context.get(step.var, None)

        chosen = None
        if val in step.cases:
            chosen = step.cases[val]
        else:
            b = _parse_bool_like(val)
            if b in step.cases:
                chosen = step.cases[b]
            elif "default" in step.cases:
                chosen = step.cases["default"]

        if chosen is None:
            self._done_cb(False, f"branch: no case for var='{step.var}' value={val!r}")
            return

        injected = [OrchestratorNode.parse_step_dict(s) for s in chosen]
        self._node.get_logger().info(
            f"[step {self._idx}] branch on '{step.var}'={val!r}: injecting {len(injected)} step(s)"
        )

        self._steps = self._steps[: self._idx + 1] + injected + self._steps[self._idx + 1 :]
        self._idx += 1
        self._run_next()

    # ---- sleep ----
    def _run_sleep(self, step: StepSpec):
        dur = float(step.duration_sec or 0.0)
        self._node.get_logger().info(f"[step {self._idx}] sleep {dur:.3f}s")

        if dur <= 0.0:
            self._idx += 1
            self._run_next()
            return

        if self._sleep_timer is not None:
            try:
                self._node.destroy_timer(self._sleep_timer)
            except Exception:
                pass
            self._sleep_timer = None

        def _fire():
            if self._sleep_timer is not None:
                try:
                    self._node.destroy_timer(self._sleep_timer)
                except Exception:
                    pass
                self._sleep_timer = None
            self._idx += 1
            self._run_next()

        self._sleep_timer = self._node.create_timer(dur, _fire)


class OrchestratorNode(Node):
    def __init__(self):
        super().__init__("inspector")

        self.declare_parameter("voice_topic", "/voice/command")
        self.declare_parameter("status_topic", "/orchestrator/status")
        self.declare_parameter("routines_file", "")
        self.declare_parameter("min_confidence", 50.0)

        self._voice_topic = self.get_parameter("voice_topic").value
        self._status_topic = self.get_parameter("status_topic").value
        self._routines_file = self.get_parameter("routines_file").value
        self._min_confidence = float(self.get_parameter("min_confidence").value)

        self._status_pub = self.create_publisher(String, self._status_topic, 10)
        self._sub = self.create_subscription(String, self._voice_topic, self._on_iros_voice_command, 10)

        self._client_cache = ServiceClientCache(self)
        self._queue = deque()
        self._busy = False

        self._routines = self._load_routines(self._routines_file)
        self.get_logger().info(f"Loaded routines: {list(self._routines.keys())}")
        self._publish_status({"state": "ready", "routines": list(self._routines.keys())})

    def _publish_status(self, payload: Dict[str, Any]):
        msg = String()
        msg.data = json.dumps(payload, ensure_ascii=False)
        self._status_pub.publish(msg)

    @staticmethod
    def parse_step_dict(s: Dict[str, Any]) -> StepSpec:
        t = str(s.get("type", "service"))

        if t == "service":
            return StepSpec(
                step_type="service",
                service_name=str(s["service_name"]),
                service_type=str(s["service_type"]),
                request=dict(s.get("request", {}) or {}),
                success_field=str(s.get("success_field", "success")),
                message_field=str(s.get("message_field", "message")),
            )

        if t == "wait_topic":
            return StepSpec(
                step_type="wait_topic",
                topic_name=str(s["topic_name"]),
                topic_type=str(s["topic_type"]),
                store_as=str(s["store_as"]),
                timeout_sec=float(s.get("timeout_sec", 0.0) or 0.0),
            )

        if t == "branch":
            return StepSpec(
                step_type="branch",
                var=str(s["var"]),
                cases=s.get("cases", {}) or {},
            )

        if t == "sleep":
            return StepSpec(
                step_type="sleep",
                duration_sec=float(s.get("duration_sec", 0.0) or 0.0),
            )

        return StepSpec(step_type=t)

    def _load_routines(self, path: str) -> Dict[str, List[StepSpec]]:
        # Default routine aligned with your current TF names seen in logs:
        # pose_up, pick_pose, pose_ok, pose_nok
        default_yaml_like = {
            "inspect": {
                "steps": [
                    {
                        "type": "service",
                        "service_name": "/go_to_frame",
                        "service_type": "iros_custom_msgs/srv/GoToFrame",
                        "request": {"frame": "pose_up"},
                    },
                    {
                        "type": "sleep",
                        "duration_sec": 0.2,
                    },
                    {
                        "type": "service",
                        "service_name": "/cv_algorithms/run",
                        "service_type": "std_srvs/srv/Trigger",
                        "request": {},
                    },
                    {
                        "type": "wait_topic",
                        "topic_name": "/cv_algorithms/result/summary",
                        "topic_type": "std_msgs/msg/Bool",
                        "store_as": "cv_ok",
                        "timeout_sec": 15.0,
                    },
                    {
                        "type": "service",
                        "service_name": "/go_to_frame",
                        "service_type": "iros_custom_msgs/srv/GoToFrame",
                        "request": {"frame": "pick_pose"},
                    },
                    {
                        "type": "sleep",
                        "duration_sec": 0.2,
                    },
                    {
                        "type": "service",
                        "service_name": "/gripper_action",
                        "service_type": "iros_custom_msgs/srv/GripperAction",
                        "request": {"open": False},
                    },
                    {
                        "type": "sleep",
                        "duration_sec": 0.2,
                    },
                    {
                        "type": "branch",
                        "var": "cv_ok",
                        "cases": {
                            True: [
                                {
                                    "type": "service",
                                    "service_name": "/go_to_frame",
                                    "service_type": "iros_custom_msgs/srv/GoToFrame",
                                    "request": {"frame": "pose_ok"},
                                }
                            ],
                            False: [
                                {
                                    "type": "service",
                                    "service_name": "/go_to_frame",
                                    "service_type": "iros_custom_msgs/srv/GoToFrame",
                                    "request": {"frame": "pose_nok"},
                                }
                            ],
                        },
                    },
                    {
                        "type": "sleep",
                        "duration_sec": 0.2,
                    },
                    {
                        "type": "service",
                        "service_name": "/gripper_action",
                        "service_type": "iros_custom_msgs/srv/GripperAction",
                        "request": {"open": True},
                    },
                ]
            }
        }

        if not path:
            return {k: [self.parse_step_dict(x) for x in v["steps"]] for k, v in default_yaml_like.items()}

        if yaml is None:
            self.get_logger().warn("PyYAML not installed; using default routines")
            return {k: [self.parse_step_dict(x) for x in v["steps"]] for k, v in default_yaml_like.items()}

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        except Exception as e:
            self.get_logger().warn(f"Failed to load routines_file='{path}': {e}; using default")
            return {k: [self.parse_step_dict(x) for x in v["steps"]] for k, v in default_yaml_like.items()}

        routines: Dict[str, List[StepSpec]] = {}
        for intent, spec in (data.get("routines", {}) or {}).items():
            raw_steps = (spec.get("steps", []) or [])
            steps = [self.parse_step_dict(s) for s in raw_steps]
            if steps:
                routines[str(intent)] = steps

        if not routines:
            return {k: [self.parse_step_dict(x) for x in v["steps"]] for k, v in default_yaml_like.items()}
        return routines

    def _on_iros_voice_command(self, msg: String):
        try:
            payload = json.loads(msg.data)
        except Exception as e:
            self.get_logger().warn(f"Bad JSON on {self._voice_topic}: {e} | data={msg.data!r}")
            return

        intent = payload.get("intent", None)
        confidence = float(payload.get("confidence", 0.0))

        if intent is None:
            self.get_logger().warn(f"Voice command missing 'intent': {payload}")
            return
        if confidence < self._min_confidence:
            self.get_logger().info(f"Ignored intent={intent} due to low confidence={confidence}")
            return

        routine = self._routines.get(str(intent))
        if not routine:
            self.get_logger().warn(f"No routine for intent='{intent}'")
            self._publish_status({"state": "rejected", "reason": "no_routine", "intent": intent, "payload": payload})
            return

        self._queue.append((str(intent), payload, list(routine)))
        self._publish_status({"state": "queued", "intent": intent, "queue_size": len(self._queue)})
        self._try_start_next()

    def _try_start_next(self):
        if self._busy or not self._queue:
            return

        intent, payload, routine = self._queue.popleft()
        self._busy = True
        self._publish_status({"state": "running", "intent": intent, "payload": payload, "steps": len(routine)})

        runner = RoutineRunner(
            node=self,
            client_cache=self._client_cache,
            steps=routine,
            context={"intent": intent, "payload": payload},
            done_cb=lambda ok, message: self._on_routine_done(intent, ok, message),
        )
        runner.start()

    def _on_routine_done(self, intent: str, ok: bool, message: str):
        self._busy = False
        self._publish_status({"state": "done", "intent": intent, "success": bool(ok), "message": message})
        self._try_start_next()


def main():
    rclpy.init()
    node = OrchestratorNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
