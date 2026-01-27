#!/usr/bin/env python3
import importlib
import json
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, List

import yaml

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

from std_msgs.msg import String


# ROS2: set_message_fields находится тут
try:
    from rosidl_runtime_py.set_message import set_message_fields  # type: ignore
except Exception:
    # fallback (упрощённый)
    def set_message_fields(msg: Any, values: Dict[str, Any], strict_mode: bool = False) -> None:
        for k, v in values.items():
            if hasattr(msg, k):
                setattr(msg, k, v)


def _import_srv(type_str: str):
    # "pkg/srv/Name"
    parts = type_str.split("/")
    if len(parts) != 3 or parts[1] != "srv":
        raise ValueError(f"Bad service_type '{type_str}', expected 'pkg/srv/Name'")
    pkg, _, name = parts
    mod = importlib.import_module(f"{pkg}.srv")
    return getattr(mod, name)


def _import_msg(type_str: str):
    # "pkg/msg/Name"
    parts = type_str.split("/")
    if len(parts) != 3 or parts[1] != "msg":
        raise ValueError(f"Bad topic_type '{type_str}', expected 'pkg/msg/Name'")
    pkg, _, name = parts
    mod = importlib.import_module(f"{pkg}.msg")
    return getattr(mod, name)


def _now() -> float:
    return time.monotonic()


def _as_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        return v.strip().lower() in ("1", "true", "yes", "y", "ok")
    return bool(v)


@dataclass
class ActiveWaitTopic:
    topic: str
    started_at: float
    timeout: float
    store_as: str


@dataclass
class ActiveServiceCall:
    service_name: str
    started_at: float
    timeout: float
    future: Any
    store_as: Optional[str] = None


@dataclass
class ActiveSleep:
    until: float


class RoutineRunner:
    def __init__(self, node: Node):
        self.node = node
        self.cb_group = ReentrantCallbackGroup()

        self.routines: Dict[str, Any] = {}
        self.ctx: Dict[str, Any] = {}

        self._steps: List[Dict[str, Any]] = []
        self._ip: int = 0
        self._running: bool = False
        self._routine_name: str = ""

        self._active_wait_topic: Optional[ActiveWaitTopic] = None
        self._active_service: Optional[ActiveServiceCall] = None
        self._active_sleep: Optional[ActiveSleep] = None
        self._waiting_until: float = 0.0

        self._clients: Dict[str, Any] = {}
        self._subs: Dict[str, Any] = {}

    def load_from_file(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        routines = data.get("routines", data)
        if not isinstance(routines, dict):
            raise ValueError("routines.yaml: expected dict at top-level or in key 'routines'")
        self.routines = routines
        self.node.get_logger().info(f"Loaded routines: {list(self.routines.keys())} (file={path})")

    @property
    def busy(self) -> bool:
        return self._running

    def start(self, name: str) -> None:
        if self._running:
            self.node.get_logger().warn(f"Busy with '{self._routine_name}', ignore new routine '{name}'")
            return

        routine = self.routines.get(name)
        if routine is None:
            self.node.get_logger().warn(f"Unknown routine '{name}'. Available: {list(self.routines.keys())}")
            return

        steps = routine.get("steps", [])
        if not isinstance(steps, list) or len(steps) == 0:
            self.node.get_logger().warn(f"Routine '{name}' has no steps")
            return

        self.ctx = {}
        self._steps = steps
        self._ip = 0
        self._running = True
        self._routine_name = name

        self._active_wait_topic = None
        self._active_service = None
        self._active_sleep = None
        self._waiting_until = 0.0

        self.node.get_logger().info(f"Start routine '{name}', steps={len(steps)}")

    def _finish_step(self) -> None:
        delay_global = float(self.node.get_parameter("post_step_delay_sec").value)
        delay_after = 0.0

        step = self._steps[self._ip]
        if isinstance(step, dict):
            delay_after = float(step.get("delay_after_sec", 0.0) or 0.0)

        self._waiting_until = _now() + delay_global + delay_after
        self._ip += 1

        self._active_wait_topic = None
        self._active_service = None
        self._active_sleep = None

    def _abort(self, reason: str) -> None:
        self.node.get_logger().error(f"Routine '{self._routine_name}' aborted: {reason}")
        self._running = False

        # cleanup wait subscriptions
        for key, sub in list(self._subs.items()):
            try:
                self.node.destroy_subscription(sub)
            except Exception:
                pass
            self._subs.pop(key, None)

        self._active_wait_topic = None
        self._active_service = None
        self._active_sleep = None

    def _get_client(self, service_name: str, srv_type: str):
        key = f"{service_name}|{srv_type}"
        if key in self._clients:
            return self._clients[key]
        srv_cls = _import_srv(srv_type)
        client = self.node.create_client(srv_cls, service_name, callback_group=self.cb_group)
        self._clients[key] = client
        return client

    def _wait_topic_cb(self, store_as: str, sub_key: str, msg: Any) -> None:
        # сохраняем "msg.data" если есть, иначе весь msg
        val = getattr(msg, "data", msg)
        self.ctx[store_as] = val
        self.node.get_logger().info(f"[wait_topic] got {sub_key}: {val}")

        sub = self._subs.pop(sub_key, None)
        if sub is not None:
            try:
                self.node.destroy_subscription(sub)
            except Exception:
                pass

        self._active_wait_topic = None
        self._finish_step()

    def tick(self) -> None:
        if not self._running:
            return

        t = _now()
        if t < self._waiting_until:
            return

        if self._ip >= len(self._steps):
            self.node.get_logger().info(f"Routine '{self._routine_name}' finished OK")
            self._running = False
            return

        step = self._steps[self._ip]
        if not isinstance(step, dict):
            self._abort(f"step {self._ip} must be dict, got {type(step)}")
            return

        stype = step.get("type")
        if stype is None:
            self._abort(f"step {self._ip} has no 'type'")
            return

        # --- ACTIVE: service ---
        if self._active_service is not None:
            if t - self._active_service.started_at > self._active_service.timeout:
                self._abort(f"service timeout: {self._active_service.service_name}")
                return
            if self._active_service.future.done():
                try:
                    resp = self._active_service.future.result()
                except Exception as e:
                    self._abort(f"service exception: {self._active_service.service_name}: {e}")
                    return

                # если есть поле success и оно False — считаем провалом (для GoToFrame/GripperAction)
                if hasattr(resp, "success") and (resp.success is False):
                    msg = getattr(resp, "message", "")
                    if bool(self.node.get_parameter("abort_on_failure").value):
                        self._abort(f"{self._active_service.service_name} returned success=False: {msg}")
                        return
                    self.node.get_logger().warn(
                        f"{self._active_service.service_name} returned success=False: {msg} (continue)"
                    )

                if self._active_service.store_as:
                    self.ctx[self._active_service.store_as] = resp

                # лог (короткий)
                m = getattr(resp, "message", None)
                if isinstance(m, str) and len(m) > 200:
                    m = m[:200] + "..."
                if m is not None:
                    self.node.get_logger().info(f"[service] {self._active_service.service_name} message={m}")

                self._finish_step()
            return

        # --- ACTIVE: wait_topic ---
        if self._active_wait_topic is not None:
            if t - self._active_wait_topic.started_at > self._active_wait_topic.timeout:
                self._abort(f"wait_topic timeout: {self._active_wait_topic.topic}")
            return

        # --- ACTIVE: sleep ---
        if self._active_sleep is not None:
            if t >= self._active_sleep.until:
                self._finish_step()
            return

        # --- START step ---
        if stype == "sleep":
            dur = float(step.get("duration_sec", 0.0))
            self.node.get_logger().info(f"[step {self._ip}] sleep {dur}s")
            self._active_sleep = ActiveSleep(until=t + dur)
            return

        if stype == "service":
            service_name = str(step["service_name"])
            srv_type = str(step["service_type"])
            req_dict = step.get("request", {}) or {}

            timeout = step.get("service_timeout_sec", None)
            if timeout is None:
                timeout = float(self.node.get_parameter("default_service_timeout_sec").value)
            timeout = float(timeout)

            store_as = step.get("store_as", None)

            client = self._get_client(service_name, srv_type)
            if not client.service_is_ready():
                # не блокируемся — ждём готовности
                # (таймаут считаем от момента "начала шага")
                self.node.get_logger().info(f"[step {self._ip}] wait service {service_name} ...")
                # имитируем активный сервис-стейт без future, чтобы отлавливать таймаут
                dummy_future = rclpy.task.Future()
                self._active_service = ActiveServiceCall(
                    service_name=service_name, started_at=t, timeout=timeout, future=dummy_future
                )
                # но future не завершится; поэтому перепишем логику: как только сервис появится — стартанём заново
                # (упрощение: сбросим active_service и попробуем снова в следующем tick)
                self._active_service = None
                self._waiting_until = t + 0.1
                return

            srv_cls = _import_srv(srv_type)
            req = srv_cls.Request()
            if isinstance(req_dict, dict) and len(req_dict) > 0:
                set_message_fields(req, req_dict, False)

            self.node.get_logger().info(f"[step {self._ip}] call {service_name} {srv_type} req={req_dict}")
            future = client.call_async(req)
            self._active_service = ActiveServiceCall(
                service_name=service_name,
                started_at=t,
                timeout=timeout,
                future=future,
                store_as=store_as,
            )
            return

        if stype == "wait_topic":
            topic = str(step["topic_name"])
            topic_type = str(step["topic_type"])
            store_as = str(step.get("store_as", "topic_value"))
            timeout = float(step.get("timeout_sec", 10.0))

            msg_cls = _import_msg(topic_type)

            sub_key = f"{topic}|{topic_type}|{store_as}"
            if sub_key not in self._subs:
                self.node.get_logger().info(f"[step {self._ip}] wait topic {topic} ({topic_type}) -> {store_as}")
                sub = self.node.create_subscription(
                    msg_cls,
                    topic,
                    lambda msg, _sa=store_as, _k=sub_key: self._wait_topic_cb(_sa, _k, msg),
                    10,
                    callback_group=self.cb_group,
                )
                self._subs[sub_key] = sub

            self._active_wait_topic = ActiveWaitTopic(topic=topic, started_at=t, timeout=timeout, store_as=store_as)
            return

        if stype == "branch":
            var = str(step["var"])
            cases = step.get("cases", {}) or {}
            val = self.ctx.get(var, False)
            b = _as_bool(val)

            # YAML "true/false" могут стать bool-ключами True/False
            selected = None
            if b in cases:
                selected = cases[b]
            else:
                selected = cases.get("true" if b else "false")

            if selected is None:
                self._abort(f"branch: no case for {b} in var '{var}'")
                return
            if not isinstance(selected, list):
                self._abort("branch: case must be list of steps")
                return

            self.node.get_logger().info(f"[step {self._ip}] branch var='{var}' -> {b}, insert {len(selected)} steps")

            # заменяем текущий branch на выбранные шаги
            self._steps = self._steps[: self._ip] + selected + self._steps[self._ip + 1 :]
            return

        self._abort(f"Unknown step type '{stype}' at step {self._ip}")


class Inspector(Node):
    def __init__(self):
        super().__init__("inspector")

        self.declare_parameter("routines_file", "")
        self.declare_parameter("min_confidence", 50.0)
        self.declare_parameter("command_topic", "/voice/command")

        self.declare_parameter("post_step_delay_sec", 0.0)          # общий delay после каждого шага
        self.declare_parameter("default_service_timeout_sec", 30.0) # если в шаге не задано
        self.declare_parameter("abort_on_failure", True)

        self.runner = RoutineRunner(self)

        routines_file = str(self.get_parameter("routines_file").value)
        if not routines_file:
            raise RuntimeError("Parameter 'routines_file' is empty")

        self.runner.load_from_file(routines_file)

        topic = str(self.get_parameter("command_topic").value)
        self.sub_cmd = self.create_subscription(String, topic, self._on_cmd, 10)

        self.timer = self.create_timer(0.05, self.runner.tick)  # 20 Hz

    def _on_cmd(self, msg: String) -> None:
        try:
            data = json.loads(msg.data)
        except Exception as e:
            self.get_logger().warn(f"Bad JSON in /voice/command: {e}")
            return

        intent = data.get("intent")
        conf = float(data.get("confidence", 0.0))

        min_conf = float(self.get_parameter("min_confidence").value)
        if conf < min_conf:
            self.get_logger().info(f"Ignore command intent='{intent}' conf={conf} < {min_conf}")
            return

        if not intent:
            self.get_logger().warn("Command has no 'intent'")
            return

        self.get_logger().info(f"Command received: intent='{intent}' conf={conf}")
        self.runner.start(str(intent))


def main():
    rclpy.init()
    node = Inspector()
    ex = MultiThreadedExecutor(num_threads=4)
    ex.add_node(node)
    try:
        ex.spin()
    finally:
        ex.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
