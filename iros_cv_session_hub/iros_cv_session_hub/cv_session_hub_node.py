#!/usr/bin/env python3
from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import qos_profile_sensor_data

from std_srvs.srv import Trigger
from std_msgs.msg import Bool, String
from sensor_msgs.msg import Image


# ---------------- small utils ----------------

@dataclass
class TimedMsg:
    msg: Any
    t_mono: float  # time.monotonic() at receipt


def _now_mono() -> float:
    return time.monotonic()


def _safe_json_loads(s: str) -> Optional[dict]:
    try:
        return json.loads(s)
    except Exception:
        return None


def _pick_bool_from_dict(d: dict, keys: Tuple[str, ...], default: Optional[bool] = None) -> Optional[bool]:
    for k in keys:
        if k in d:
            v = d[k]
            if isinstance(v, bool):
                return v
            if isinstance(v, (int, float)):
                return bool(v)
            if isinstance(v, str):
                vl = v.strip().lower()
                if vl in ("true", "1", "yes", "y", "ok", "pass", "passed", "success"):
                    return True
                if vl in ("false", "0", "no", "n", "fail", "failed", "error"):
                    return False
    return default


def _pick_float_from_dict(d: dict, keys: Tuple[str, ...], default: Optional[float] = None) -> Optional[float]:
    for k in keys:
        if k in d:
            v = d[k]
            if isinstance(v, (int, float)):
                return float(v)
            if isinstance(v, str):
                try:
                    return float(v.strip())
                except Exception:
                    pass
    return default


def _pick_int_from_dict(d: dict, keys: Tuple[str, ...], default: Optional[int] = None) -> Optional[int]:
    for k in keys:
        if k in d:
            v = d[k]
            if isinstance(v, bool):
                return int(v)
            if isinstance(v, int):
                return int(v)
            if isinstance(v, float):
                return int(v)
            if isinstance(v, str):
                try:
                    return int(float(v.strip()))
                except Exception:
                    pass
    return default


# ---------------- node ----------------

class CvSessionHubNode(Node):
    """
    Hub node with 3 independent "run" triggers + 1 "publish" trigger (one-shot).

    Updated rust parsing for NEW rust node JSON:
      - rust classification (ResNet decision): d["run_unet"] (bool)
      - final detection (UNet+post): d["detected"] (bool)
      - rust probability: d["rust_prob"] (float)
    Select which one should be treated as "rust_detected" via param rust_detect_mode:
      - "resnet" -> use run_unet as rust_detected (your desired behavior)
      - "final"  -> use detected as rust_detected
    Default is "resnet".

    Additional:
    - On ~/publish, publishes /<output_prefix>/ok as Bool for a short burst (few seconds),
      so downstream nodes can catch it.
        ok = (rust_detected == False) AND (pcb_ok == True) AND (gear_ok == True)
    """

    def __init__(self) -> None:
        super().__init__("iros_cv_session_hub")

        self._cbg = ReentrantCallbackGroup()

        # -------- params --------
        self.declare_parameter("listen_duration_s", 2.0)
        self.declare_parameter("output_prefix", "/cv_hub")

        # ok burst publishing
        self.declare_parameter("ok_burst_duration_s", 2.0)  # publish /ok for this many seconds
        self.declare_parameter("ok_burst_rate_hz", 10.0)    # publish frequency during burst

        # rust inputs
        self.declare_parameter("rust_service", "/rust_detect/run")          # std_srvs/Trigger
        self.declare_parameter("rust_service_timeout_s", 2.0)
        self.declare_parameter("rust_prob_topic", "/rust/prob")             # sensor_msgs/Image
        self.declare_parameter("rust_detected_topic", "/rust/detected")     # std_msgs/Bool (optional)
        self.declare_parameter("rust_detect_mode", "resnet")                # "resnet"|"final"

        # pcb inputs
        self.declare_parameter("pcb_infer_service", "/pcb_inspector/inference")   # std_srvs/Trigger
        self.declare_parameter("pcb_service_timeout_s", 3.0)
        self.declare_parameter("pcb_report_topic", "/pcb_inspector/report")       # std_msgs/String(JSON)
        self.declare_parameter("pcb_annotated_topic", "/pcb_inspector/annotated") # sensor_msgs/Image

        # gear inputs
        self.declare_parameter("gear_infer_service", "/gear_inspector/inference")   # std_srvs/Trigger
        self.declare_parameter("gear_service_timeout_s", 3.0)
        self.declare_parameter("gear_report_topic", "/gear_inspector/report")       # std_msgs/String(JSON)
        self.declare_parameter("gear_annotated_topic", "/gear_inspector/annotated") # sensor_msgs/Image

        # -------- locks/state --------
        self._live_lock = threading.Lock()
        self._snap_lock = threading.Lock()

        self._run_lock_rust = threading.Lock()
        self._run_lock_pcb = threading.Lock()
        self._run_lock_gear = threading.Lock()

        # ok burst state
        self._ok_burst_lock = threading.Lock()
        self._ok_burst_timer = None
        self._ok_burst_end_mono: float = 0.0
        self._ok_burst_value: bool = False

        # live buffers
        self._live_rust_prob: Optional[TimedMsg] = None
        self._live_rust_detected: Optional[TimedMsg] = None

        self._live_pcb_annot: Optional[TimedMsg] = None
        self._live_pcb_report: Optional[TimedMsg] = None

        self._live_gear_annot: Optional[TimedMsg] = None
        self._live_gear_report: Optional[TimedMsg] = None

        # snapshots: rust
        self._snap_rust_prob: Optional[Image] = None
        self._snap_rust_detected: Optional[bool] = None  # per rust_detect_mode
        self._snap_rust_service: Optional[dict] = None
        self._snap_rust_report_str: str = ""
        self._snap_rust_meta: Dict[str, Any] = {}
        self._snap_rust_last_ts_unix: Optional[float] = None

        # snapshots: pcb
        self._snap_pcb_annot: Optional[Image] = None
        self._snap_pcb_ok: Optional[bool] = None
        self._snap_pcb_report: Optional[dict] = None
        self._snap_pcb_report_str: str = ""
        self._snap_pcb_meta: Dict[str, Any] = {}
        self._snap_pcb_last_ts_unix: Optional[float] = None

        # snapshots: gear
        self._snap_gear_annot: Optional[Image] = None
        self._snap_gear_ok: Optional[bool] = None
        self._snap_gear_report: Optional[dict] = None
        self._snap_gear_report_str: str = ""
        self._snap_gear_meta: Dict[str, Any] = {}
        self._snap_gear_last_ts_unix: Optional[float] = None

        self._snap_combined_report_str: str = ""

        # -------- I/O --------
        out_prefix = str(self.get_parameter("output_prefix").value).rstrip("/") or "/cv_hub"

        # Publishers (one-shot publish via ~/publish)
        self._pub_rust_prob = self.create_publisher(Image, f"{out_prefix}/rust/prob", qos_profile_sensor_data)
        self._pub_rust_det = self.create_publisher(Bool, f"{out_prefix}/rust/detected", 10)
        self._pub_rust_rep = self.create_publisher(String, f"{out_prefix}/rust/report", 10)

        self._pub_pcb_annot = self.create_publisher(Image, f"{out_prefix}/pcb/annotated", qos_profile_sensor_data)
        self._pub_pcb_ok = self.create_publisher(Bool, f"{out_prefix}/pcb/ok", 10)
        self._pub_pcb_rep = self.create_publisher(String, f"{out_prefix}/pcb/report", 10)

        self._pub_gear_annot = self.create_publisher(Image, f"{out_prefix}/gear/annotated", qos_profile_sensor_data)
        self._pub_gear_ok = self.create_publisher(Bool, f"{out_prefix}/gear/ok", 10)
        self._pub_gear_rep = self.create_publisher(String, f"{out_prefix}/gear/report", 10)

        self._pub_report = self.create_publisher(String, f"{out_prefix}/report", 10)
        self._pub_ok = self.create_publisher(Bool, f"{out_prefix}/ok", 10)

        # Subscribers
        self.create_subscription(
            Image, str(self.get_parameter("rust_prob_topic").value), self._on_rust_prob,
            qos_profile_sensor_data, callback_group=self._cbg
        )
        self.create_subscription(
            Bool, str(self.get_parameter("rust_detected_topic").value), self._on_rust_detected,
            10, callback_group=self._cbg
        )
        self.create_subscription(
            Image, str(self.get_parameter("pcb_annotated_topic").value), self._on_pcb_annot,
            qos_profile_sensor_data, callback_group=self._cbg
        )
        self.create_subscription(
            String, str(self.get_parameter("pcb_report_topic").value), self._on_pcb_report,
            10, callback_group=self._cbg
        )
        self.create_subscription(
            Image, str(self.get_parameter("gear_annotated_topic").value), self._on_gear_annot,
            qos_profile_sensor_data, callback_group=self._cbg
        )
        self.create_subscription(
            String, str(self.get_parameter("gear_report_topic").value), self._on_gear_report,
            10, callback_group=self._cbg
        )

        # Service clients
        self._rust_cli = self.create_client(Trigger, str(self.get_parameter("rust_service").value), callback_group=self._cbg)
        self._pcb_cli = self.create_client(Trigger, str(self.get_parameter("pcb_infer_service").value), callback_group=self._cbg)
        self._gear_cli = self.create_client(Trigger, str(self.get_parameter("gear_infer_service").value), callback_group=self._cbg)

        # Service servers
        self.create_service(Trigger, "~/run_rust", self._on_run_rust, callback_group=self._cbg)
        self.create_service(Trigger, "~/run_pcb", self._on_run_pcb, callback_group=self._cbg)
        self.create_service(Trigger, "~/run_gear", self._on_run_gear, callback_group=self._cbg)
        self.create_service(Trigger, "~/publish", self._on_publish, callback_group=self._cbg)

        self.get_logger().info(
            f"{self.get_name()} ready. Services: ~/(run_rust, run_pcb, run_gear, publish). Outputs prefix: {out_prefix}"
        )

    # -------- input callbacks --------

    def _on_rust_prob(self, msg: Image) -> None:
        with self._live_lock:
            self._live_rust_prob = TimedMsg(msg=msg, t_mono=_now_mono())

    def _on_rust_detected(self, msg: Bool) -> None:
        with self._live_lock:
            self._live_rust_detected = TimedMsg(msg=msg, t_mono=_now_mono())

    def _on_pcb_annot(self, msg: Image) -> None:
        with self._live_lock:
            self._live_pcb_annot = TimedMsg(msg=msg, t_mono=_now_mono())

    def _on_pcb_report(self, msg: String) -> None:
        with self._live_lock:
            self._live_pcb_report = TimedMsg(msg=msg, t_mono=_now_mono())

    def _on_gear_annot(self, msg: Image) -> None:
        with self._live_lock:
            self._live_gear_annot = TimedMsg(msg=msg, t_mono=_now_mono())

    def _on_gear_report(self, msg: String) -> None:
        with self._live_lock:
            self._live_gear_report = TimedMsg(msg=msg, t_mono=_now_mono())

    # -------- helpers --------

    def _call_trigger(self, client: rclpy.client.Client, timeout_s: float) -> Tuple[bool, Optional[Trigger.Response], str]:
        if not client.service_is_ready():
            if not client.wait_for_service(timeout_sec=timeout_s):
                return False, None, "service_not_available"

        fut = client.call_async(Trigger.Request())
        ev = threading.Event()
        fut.add_done_callback(lambda _f: ev.set())

        if not ev.wait(timeout=timeout_s):
            return False, None, "service_call_timeout"

        try:
            return True, fut.result(), ""
        except Exception as e:
            return True, None, f"service_call_error:{e}"

    def _sleep_listen_window(self, listen_s: float) -> None:
        end_t = _now_mono() + listen_s
        while _now_mono() < end_t:
            time.sleep(0.02)

    def _copy_live(self) -> Dict[str, Optional[TimedMsg]]:
        with self._live_lock:
            return {
                "rust_prob": self._live_rust_prob,
                "rust_det": self._live_rust_detected,
                "pcb_annot": self._live_pcb_annot,
                "pcb_report": self._live_pcb_report,
                "gear_annot": self._live_gear_annot,
                "gear_report": self._live_gear_report,
            }

    @staticmethod
    def _is_fresh(tm: Optional[TimedMsg], window_start_mono: float) -> bool:
        return bool(tm and tm.t_mono >= window_start_mono)

    def _compose_check_report(
        self,
        check_name: str,
        service_ok: bool,
        service_err: str,
        service_resp_success: Optional[bool],
        service_resp_message_dict: Optional[dict],
        topic_meta: Dict[str, Any],
        result_fields: Dict[str, Any],
    ) -> str:
        out = {
            "check": check_name,
            "service": {
                "called": True,
                "call_ok": bool(service_ok),
                "error": service_err,
                "resp_success": service_resp_success,
                "resp_data": service_resp_message_dict,
            },
            "topics": topic_meta,
            "result": result_fields,
            "ts_unix": time.time(),
        }
        return json.dumps(out, ensure_ascii=False)

    @staticmethod
    def _compute_global_ok(rust_detected: Optional[bool], pcb_ok: Optional[bool], gear_ok: Optional[bool]) -> bool:
        return (rust_detected is False) and (pcb_ok is True) and (gear_ok is True)

    def _start_ok_burst(self, value: bool) -> None:
        duration_s = float(self.get_parameter("ok_burst_duration_s").value) or 0.0
        rate_hz = float(self.get_parameter("ok_burst_rate_hz").value) or 0.0
        if duration_s <= 0.0 or rate_hz <= 0.0:
            m = Bool()
            m.data = bool(value)
            self._pub_ok.publish(m)
            return

        period_s = 1.0 / max(1e-6, rate_hz)

        with self._ok_burst_lock:
            if self._ok_burst_timer is not None:
                self._ok_burst_timer.cancel()
                self._ok_burst_timer = None

            self._ok_burst_value = bool(value)
            self._ok_burst_end_mono = _now_mono() + duration_s

            m = Bool()
            m.data = self._ok_burst_value
            self._pub_ok.publish(m)

            def _tick():
                with self._ok_burst_lock:
                    if _now_mono() >= self._ok_burst_end_mono:
                        if self._ok_burst_timer is not None:
                            self._ok_burst_timer.cancel()
                            self._ok_burst_timer = None
                        return
                    v = self._ok_burst_value

                mm = Bool()
                mm.data = v
                self._pub_ok.publish(mm)

            self._ok_burst_timer = self.create_timer(period_s, _tick, callback_group=self._cbg)

    def _compose_combined_report_locked(self) -> str:
        global_ok = self._compute_global_ok(self._snap_rust_detected, self._snap_pcb_ok, self._snap_gear_ok)
        out = {
            "ok": global_ok,
            "rust": {
                "detected": self._snap_rust_detected,
                "service_data": self._snap_rust_service,
                "meta": self._snap_rust_meta,
                "last_ts_unix": self._snap_rust_last_ts_unix,
                "report": _safe_json_loads(self._snap_rust_report_str) if self._snap_rust_report_str else None,
            },
            "pcb": {
                "ok": self._snap_pcb_ok,
                "report_data": self._snap_pcb_report,
                "meta": self._snap_pcb_meta,
                "last_ts_unix": self._snap_pcb_last_ts_unix,
                "report": _safe_json_loads(self._snap_pcb_report_str) if self._snap_pcb_report_str else None,
            },
            "gear": {
                "ok": self._snap_gear_ok,
                "report_data": self._snap_gear_report,
                "meta": self._snap_gear_meta,
                "last_ts_unix": self._snap_gear_last_ts_unix,
                "report": _safe_json_loads(self._snap_gear_report_str) if self._snap_gear_report_str else None,
            },
            "ts_unix": time.time(),
        }
        return json.dumps(out, ensure_ascii=False)

    # -------- services: run_* --------

    def _on_run_rust(self, _req: Trigger.Request, res: Trigger.Response) -> Trigger.Response:
        if not self._run_lock_rust.acquire(blocking=False):
            res.success = False
            res.message = json.dumps({"error": "busy_rust"}, separators=(",", ":"))
            return res

        try:
            listen_s = float(self.get_parameter("listen_duration_s").value) or 2.0
            timeout_s = float(self.get_parameter("rust_service_timeout_s").value) or 2.0
            detect_mode = str(self.get_parameter("rust_detect_mode").value).strip().lower() or "resnet"

            window_start = _now_mono()
            call_ok, call_resp, call_err = self._call_trigger(self._rust_cli, timeout_s)
            self._sleep_listen_window(listen_s)

            live = self._copy_live()
            got_prob = self._is_fresh(live["rust_prob"], window_start)
            got_det_topic = self._is_fresh(live["rust_det"], window_start)

            rust_prob_msg = live["rust_prob"].msg if got_prob and live["rust_prob"] else None
            rust_det_from_topic = bool(live["rust_det"].msg.data) if got_det_topic and live["rust_det"] else None

            rust_service_dict: Optional[dict] = None
            resp_success: Optional[bool] = None

            rust_final: Optional[bool] = None      # UNet+post: d["detected"]
            rust_resnet: Optional[bool] = None     # ResNet decision: d["run_unet"]
            rust_prob_val: Optional[float] = None  # d["rust_prob"]

            rust_detected: Optional[bool] = None   # selected per mode

            if call_resp is not None:
                resp_success = bool(getattr(call_resp, "success", False))
                msg = str(getattr(call_resp, "message", ""))
                d = _safe_json_loads(msg)

                if isinstance(d, dict):
                    rust_service_dict = d

                    rust_final = _pick_bool_from_dict(d, ("detected", "rust_detected", "final_detected"), default=None)
                    rust_resnet = _pick_bool_from_dict(d, ("run_unet", "resnet_detected", "rust_pred"), default=None)
                    rust_prob_val = _pick_float_from_dict(d, ("rust_prob", "prob", "score"), default=None)

                    if detect_mode == "final":
                        rust_detected = rust_final if rust_final is not None else rust_resnet
                    else:
                        # default: resnet
                        rust_detected = rust_resnet if rust_resnet is not None else rust_final

                    if rust_detected is None:
                        rust_detected = resp_success
                else:
                    rust_service_dict = {"raw_message": msg}
                    rust_detected = resp_success
            else:
                rust_service_dict = None
                rust_detected = None

            # fallback to topic if still unknown
            if rust_detected is None:
                rust_detected = rust_det_from_topic

            # detected source
            detected_source = None
            if rust_detected is not None:
                if call_resp is not None and isinstance(rust_service_dict, dict) and rust_service_dict is not None:
                    if "detected" in rust_service_dict or "run_unet" in rust_service_dict:
                        detected_source = f"service_json:{detect_mode}"
                    else:
                        detected_source = "service_unknown_json"
                elif call_resp is not None:
                    detected_source = "service_success_fallback"
                elif got_det_topic:
                    detected_source = "topic"
                else:
                    detected_source = "unknown"

            topic_meta = {"got_rust_prob": bool(got_prob), "got_rust_detected_topic": bool(got_det_topic)}
            meta = {
                "listen_duration_s": listen_s,
                "service_timeout_s": timeout_s,
                "service_error": call_err,
                "service_call_ok": bool(call_ok),
                "rust_detect_mode": detect_mode,
                **topic_meta,
            }

            rust_report_str = self._compose_check_report(
                "rust",
                call_ok, call_err, resp_success,
                rust_service_dict,
                topic_meta,
                {
                    "detected": rust_detected,
                    "detected_source": detected_source,
                    "service_final_detected": rust_final,
                    "service_resnet_detected": rust_resnet,
                    "service_rust_prob": rust_prob_val,
                },
            )

            with self._snap_lock:
                if rust_prob_msg is not None:
                    self._snap_rust_prob = rust_prob_msg
                self._snap_rust_detected = rust_detected if rust_detected is not None else None
                self._snap_rust_service = rust_service_dict
                self._snap_rust_report_str = rust_report_str
                self._snap_rust_meta = meta
                self._snap_rust_last_ts_unix = time.time()

            have_any = bool(call_resp is not None or got_prob or got_det_topic)
            res.success = bool(call_ok and have_any)
            res.message = rust_report_str
            return res

        finally:
            self._run_lock_rust.release()

    def _on_run_pcb(self, _req: Trigger.Request, res: Trigger.Response) -> Trigger.Response:
        if not self._run_lock_pcb.acquire(blocking=False):
            res.success = False
            res.message = json.dumps({"error": "busy_pcb"}, separators=(",", ":"))
            return res

        try:
            listen_s = float(self.get_parameter("listen_duration_s").value) or 2.0
            timeout_s = float(self.get_parameter("pcb_service_timeout_s").value) or 3.0

            window_start = _now_mono()
            call_ok, call_resp, call_err = self._call_trigger(self._pcb_cli, timeout_s)
            self._sleep_listen_window(listen_s)

            live = self._copy_live()
            got_annot = self._is_fresh(live["pcb_annot"], window_start)
            got_report = self._is_fresh(live["pcb_report"], window_start)

            pcb_annot_msg = live["pcb_annot"].msg if got_annot and live["pcb_annot"] else None
            pcb_report_msg = live["pcb_report"].msg if got_report and live["pcb_report"] else None

            pcb_report_dict = None
            pcb_ok = None
            if pcb_report_msg is not None:
                pcb_report_dict = _safe_json_loads(str(pcb_report_msg.data))
                if isinstance(pcb_report_dict, dict):
                    pcb_ok = _pick_bool_from_dict(pcb_report_dict, ("overall_ok", "ok", "success"), default=None)

            resp_success = bool(getattr(call_resp, "success", False)) if call_resp is not None else None
            svc_msg_dict = _safe_json_loads(str(getattr(call_resp, "message", ""))) if call_resp is not None else None
            if call_resp is not None and not isinstance(svc_msg_dict, dict):
                svc_msg_dict = {"raw_message": str(getattr(call_resp, "message", ""))}

            topic_meta = {"got_pcb_annotated": bool(got_annot), "got_pcb_report": bool(got_report)}
            meta = {
                "listen_duration_s": listen_s,
                "service_timeout_s": timeout_s,
                "service_error": call_err,
                "service_call_ok": bool(call_ok),
                **topic_meta,
            }

            pcb_report_str = self._compose_check_report(
                "pcb",
                call_ok, call_err, resp_success,
                svc_msg_dict,
                topic_meta,
                {"ok": pcb_ok},
            )

            with self._snap_lock:
                if pcb_annot_msg is not None:
                    self._snap_pcb_annot = pcb_annot_msg
                self._snap_pcb_ok = pcb_ok if pcb_ok is not None else None
                self._snap_pcb_report = pcb_report_dict if isinstance(pcb_report_dict, dict) else None
                self._snap_pcb_report_str = pcb_report_str
                self._snap_pcb_meta = meta
                self._snap_pcb_last_ts_unix = time.time()

            have_any = bool(call_resp is not None or got_annot or got_report)
            res.success = bool(call_ok and have_any)
            res.message = pcb_report_str
            return res

        finally:
            self._run_lock_pcb.release()

    def _on_run_gear(self, _req: Trigger.Request, res: Trigger.Response) -> Trigger.Response:
        if not self._run_lock_gear.acquire(blocking=False):
            res.success = False
            res.message = json.dumps({"error": "busy_gear"}, separators=(",", ":"))
            return res

        try:
            listen_s = float(self.get_parameter("listen_duration_s").value) or 2.0
            timeout_s = float(self.get_parameter("gear_service_timeout_s").value) or 3.0

            window_start = _now_mono()
            call_ok, call_resp, call_err = self._call_trigger(self._gear_cli, timeout_s)
            self._sleep_listen_window(listen_s)

            live = self._copy_live()
            got_annot = self._is_fresh(live["gear_annot"], window_start)
            got_report = self._is_fresh(live["gear_report"], window_start)

            gear_annot_msg = live["gear_annot"].msg if got_annot and live["gear_annot"] else None
            gear_report_msg = live["gear_report"].msg if got_report and live["gear_report"] else None

            gear_report_dict = None
            gear_ok = None
            if gear_report_msg is not None:
                gear_report_dict = _safe_json_loads(str(gear_report_msg.data))
                if isinstance(gear_report_dict, dict):
                    gear_ok = _pick_bool_from_dict(gear_report_dict, ("overall_ok", "ok", "success"), default=None)

            resp_success = bool(getattr(call_resp, "success", False)) if call_resp is not None else None
            svc_msg_dict = _safe_json_loads(str(getattr(call_resp, "message", ""))) if call_resp is not None else None
            if call_resp is not None and not isinstance(svc_msg_dict, dict):
                svc_msg_dict = {"raw_message": str(getattr(call_resp, "message", ""))}

            topic_meta = {"got_gear_annotated": bool(got_annot), "got_gear_report": bool(got_report)}
            meta = {
                "listen_duration_s": listen_s,
                "service_timeout_s": timeout_s,
                "service_error": call_err,
                "service_call_ok": bool(call_ok),
                **topic_meta,
            }

            gear_report_str = self._compose_check_report(
                "gear",
                call_ok, call_err, resp_success,
                svc_msg_dict,
                topic_meta,
                {"ok": gear_ok},
            )

            with self._snap_lock:
                if gear_annot_msg is not None:
                    self._snap_gear_annot = gear_annot_msg
                self._snap_gear_ok = gear_ok if gear_ok is not None else None
                self._snap_gear_report = gear_report_dict if isinstance(gear_report_dict, dict) else None
                self._snap_gear_report_str = gear_report_str
                self._snap_gear_meta = meta
                self._snap_gear_last_ts_unix = time.time()

            have_any = bool(call_resp is not None or got_annot or got_report)
            res.success = bool(call_ok and have_any)
            res.message = gear_report_str
            return res

        finally:
            self._run_lock_gear.release()

    # -------- publish (one-shot) --------

    def _publish_snapshot_once(self) -> Tuple[str, bool]:
        with self._snap_lock:
            rust_prob = self._snap_rust_prob
            rust_det = self._snap_rust_detected
            rust_rep = self._snap_rust_report_str

            pcb_annot = self._snap_pcb_annot
            pcb_ok = self._snap_pcb_ok
            pcb_rep = self._snap_pcb_report_str

            gear_annot = self._snap_gear_annot
            gear_ok = self._snap_gear_ok
            gear_rep = self._snap_gear_report_str

            global_ok = self._compute_global_ok(rust_det, pcb_ok, gear_ok)
            combined = self._compose_combined_report_locked()
            self._snap_combined_report_str = combined

        if rust_prob is not None:
            self._pub_rust_prob.publish(rust_prob)
        if rust_det is not None:
            m = Bool()
            m.data = bool(rust_det)
            self._pub_rust_det.publish(m)
        if rust_rep:
            s = String()
            s.data = rust_rep
            self._pub_rust_rep.publish(s)

        if pcb_annot is not None:
            self._pub_pcb_annot.publish(pcb_annot)
        if pcb_ok is not None:
            m = Bool()
            m.data = bool(pcb_ok)
            self._pub_pcb_ok.publish(m)
        if pcb_rep:
            s = String()
            s.data = pcb_rep
            self._pub_pcb_rep.publish(s)

        if gear_annot is not None:
            self._pub_gear_annot.publish(gear_annot)
        if gear_ok is not None:
            m = Bool()
            m.data = bool(gear_ok)
            self._pub_gear_ok.publish(m)
        if gear_rep:
            s = String()
            s.data = gear_rep
            self._pub_gear_rep.publish(s)

        self._start_ok_burst(global_ok)

        s = String()
        s.data = combined
        self._pub_report.publish(s)

        return combined, global_ok

    def _on_publish(self, _req: Trigger.Request, res: Trigger.Response) -> Trigger.Response:
        combined, _global_ok = self._publish_snapshot_once()

        d = _safe_json_loads(combined) or {}
        have_any = bool(
            (d.get("rust", {}).get("last_ts_unix") is not None)
            or (d.get("pcb", {}).get("last_ts_unix") is not None)
            or (d.get("gear", {}).get("last_ts_unix") is not None)
        )

        res.success = bool(have_any)
        res.message = combined
        return res


def main() -> None:
    rclpy.init()
    node = CvSessionHubNode()

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