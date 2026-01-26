#!/usr/bin/env python3
from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from typing import Optional, Tuple, Any, Dict

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import qos_profile_sensor_data

from std_srvs.srv import Trigger
from std_msgs.msg import Bool, String
from sensor_msgs.msg import Image


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


def _pick_bool_from_dict(d: dict, keys: Tuple[str, ...], default: bool = False) -> bool:
    for k in keys:
        if k in d:
            v = d[k]
            if isinstance(v, bool):
                return v
            if isinstance(v, (int, float)):
                return bool(v)
            if isinstance(v, str):
                if v.lower() in ("true", "1", "yes", "y", "ok"):
                    return True
                if v.lower() in ("false", "0", "no", "n", "fail"):
                    return False
    return default


class CvSessionHubNode(Node):
    """
    Session hub:
    - Trigger service ~/start opens a listening window for listen_duration_s
    - optionally calls rust + pcb Trigger services to force fresh outputs
    - stores last messages that arrived within the window
    - continuously republishes stored snapshots to output topics
    """

    def __init__(self) -> None:
        super().__init__("cv_session_hub")

        self._cbg = ReentrantCallbackGroup()

        # ---------------- params ----------------
        self.declare_parameter("listen_duration_s", 2.0)

        # Inputs: rust
        self.declare_parameter("rust_service", "/rust_detect/run")      # std_srvs/Trigger
        self.declare_parameter("rust_prob_topic", "/rust/prob")         # sensor_msgs/Image
        self.declare_parameter("rust_detected_topic", "/rust/detected") # std_msgs/Bool (optional)
        self.declare_parameter("call_rust_service", True)
        self.declare_parameter("rust_service_timeout_s", 2.0)

        # Inputs: pcb
        self.declare_parameter("pcb_infer_service", "/pcb_inspector/inference")   # std_srvs/Trigger
        self.declare_parameter("pcb_report_topic", "/pcb_inspector/report")      # std_msgs/String(JSON)
        self.declare_parameter("pcb_annotated_topic", "/pcb_inspector/annotated")# sensor_msgs/Image
        self.declare_parameter("call_pcb_infer_service", True)
        self.declare_parameter("pcb_service_timeout_s", 3.0)

        # Outputs
        self.declare_parameter("output_prefix", "/cv_hub")
        self.declare_parameter("republish_rate_hz", 5.0)

        # ---------------- state ----------------
        self._run_lock = threading.Lock()

        self._live_lock = threading.Lock()
        self._live_rust_prob: Optional[TimedMsg] = None
        self._live_rust_detected: Optional[TimedMsg] = None
        self._live_pcb_annot: Optional[TimedMsg] = None
        self._live_pcb_report: Optional[TimedMsg] = None

        self._snap_lock = threading.Lock()
        self._snap_rust_prob: Optional[Image] = None
        self._snap_pcb_annot: Optional[Image] = None
        self._snap_rust_detected: Optional[bool] = None
        self._snap_pcb_ok: Optional[bool] = None
        self._snap_report_str: str = ""

        self._last_rust_service: Optional[dict] = None
        self._last_pcb_report: Optional[dict] = None
        self._last_session_summary: Optional[dict] = None

        # ---------------- I/O ----------------
        out_prefix = str(self.get_parameter("output_prefix").value).rstrip("/")
        if out_prefix == "":
            out_prefix = "/cv_hub"

        self._pub_rust_prob = self.create_publisher(Image, f"{out_prefix}/rust_prob", qos_profile_sensor_data)
        self._pub_pcb_annot = self.create_publisher(Image, f"{out_prefix}/pcb_annotated", qos_profile_sensor_data)
        self._pub_rust_det = self.create_publisher(Bool, f"{out_prefix}/rust_detected", 10)
        self._pub_pcb_ok = self.create_publisher(Bool, f"{out_prefix}/pcb_ok", 10)
        self._pub_report = self.create_publisher(String, f"{out_prefix}/report", 10)

        # Subscribers
        rust_prob_topic = str(self.get_parameter("rust_prob_topic").value)
        rust_det_topic = str(self.get_parameter("rust_detected_topic").value)
        pcb_annot_topic = str(self.get_parameter("pcb_annotated_topic").value)
        pcb_report_topic = str(self.get_parameter("pcb_report_topic").value)

        self.create_subscription(Image, rust_prob_topic, self._on_rust_prob, qos_profile_sensor_data, callback_group=self._cbg)
        self.create_subscription(Bool, rust_det_topic, self._on_rust_detected, 10, callback_group=self._cbg)
        self.create_subscription(Image, pcb_annot_topic, self._on_pcb_annot, qos_profile_sensor_data, callback_group=self._cbg)
        self.create_subscription(String, pcb_report_topic, self._on_pcb_report, 10, callback_group=self._cbg)

        # Service clients
        self._rust_cli = self.create_client(Trigger, str(self.get_parameter("rust_service").value), callback_group=self._cbg)
        self._pcb_cli = self.create_client(Trigger, str(self.get_parameter("pcb_infer_service").value), callback_group=self._cbg)

        # Start service
        self._srv_start = self.create_service(Trigger, "~/start", self._on_start, callback_group=self._cbg)

        # Republish timer
        hz = float(self.get_parameter("republish_rate_hz").value)
        if hz <= 0:
            hz = 5.0
        self._timer = self.create_timer(1.0 / hz, self._on_republish_timer, callback_group=self._cbg)

        self.get_logger().info(
            f"cv_session_hub ready. start service: {self.get_name()}/start "
            f"outputs: {out_prefix}/(rust_prob, pcb_annotated, rust_detected, pcb_ok, report)"
        )

    # ---------------- callbacks: inputs ----------------

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

    # ---------------- helpers ----------------

    def _call_trigger(self, client: rclpy.client.Client, timeout_s: float) -> Tuple[bool, Optional[Trigger.Response], str]:
        if not client.service_is_ready():
            if not client.wait_for_service(timeout_sec=timeout_s):
                return False, None, "service_not_available"

        req = Trigger.Request()
        fut = client.call_async(req)

        ev = threading.Event()

        def _done(_f):
            ev.set()

        fut.add_done_callback(_done)
        ok = ev.wait(timeout=timeout_s)
        if not ok:
            return False, None, "service_call_timeout"

        try:
            resp = fut.result()
            return True, resp, ""
        except Exception as e:
            return True, None, f"service_call_error:{e}"

    def _snapshot_from_live(self, window_start_mono: float) -> Dict[str, Any]:
        with self._live_lock:
            live_rust_prob = self._live_rust_prob
            live_rust_det = self._live_rust_detected
            live_pcb_annot = self._live_pcb_annot
            live_pcb_report = self._live_pcb_report

        got_rust_prob = bool(live_rust_prob and live_rust_prob.t_mono >= window_start_mono)
        got_pcb_annot = bool(live_pcb_annot and live_pcb_annot.t_mono >= window_start_mono)
        got_pcb_report = bool(live_pcb_report and live_pcb_report.t_mono >= window_start_mono)
        got_rust_det_topic = bool(live_rust_det and live_rust_det.t_mono >= window_start_mono)

        rust_prob_msg = live_rust_prob.msg if got_rust_prob else None
        pcb_annot_msg = live_pcb_annot.msg if got_pcb_annot else None
        pcb_report_msg = live_pcb_report.msg if got_pcb_report else None

        pcb_report_dict = None
        pcb_ok = None
        if pcb_report_msg is not None:
            pcb_report_dict = _safe_json_loads(pcb_report_msg.data)
            if isinstance(pcb_report_dict, dict):
                pcb_ok = _pick_bool_from_dict(pcb_report_dict, ("overall_ok", "ok", "success"), default=False)

        rust_det_from_topic = None
        if got_rust_det_topic and live_rust_det is not None:
            rust_det_from_topic = bool(getattr(live_rust_det.msg, "data", False))

        return {
            "got_rust_prob": got_rust_prob,
            "got_pcb_annot": got_pcb_annot,
            "got_pcb_report": got_pcb_report,
            "got_rust_det_topic": got_rust_det_topic,
            "rust_prob_msg": rust_prob_msg,
            "pcb_annot_msg": pcb_annot_msg,
            "pcb_report_msg": pcb_report_msg,
            "pcb_report_dict": pcb_report_dict,
            "pcb_ok": pcb_ok,
            "rust_det_from_topic": rust_det_from_topic,
        }

    def _compose_combined_report(
        self,
        rust_service_dict: Optional[dict],
        pcb_report_dict: Optional[dict],
        rust_detected: Optional[bool],
        pcb_ok: Optional[bool],
        meta: Dict[str, Any],
    ) -> str:
        rust_part = {"detected": rust_detected, "data": rust_service_dict}
        pcb_part = {"ok": pcb_ok, "data": pcb_report_dict}

        rust_conf = None
        rust_bbox = None
        if isinstance(rust_service_dict, dict):
            rust_conf = rust_service_dict.get("conf", None)
            rust_bbox = rust_service_dict.get("bbox_xyxy", None)

        pcb_reason = None
        pcb_cmd = None
        if isinstance(pcb_report_dict, dict):
            pcb_reason = pcb_report_dict.get("reason", None)
            pcb_cmd = pcb_report_dict.get("command", None)

        summary_lines = []
        summary_lines.append(f"[RUST] detected={rust_detected} conf={rust_conf} bbox={rust_bbox}")
        summary_lines.append(f"[PCB] ok={pcb_ok} command={pcb_cmd} reason={pcb_reason}")
        summary_lines.append(
            f"[META] got_rust_prob={meta.get('got_rust_prob')} got_pcb_annot={meta.get('got_pcb_annot')} got_pcb_report={meta.get('got_pcb_report')}"
        )

        out = {
            "rust": rust_part,
            "pcb": pcb_part,
            "meta": meta,
            "summary": "\n".join(summary_lines),
            "ts_unix": time.time(),
        }
        return json.dumps(out, ensure_ascii=False)

    # ---------------- service: start session ----------------

    def _on_start(self, _req: Trigger.Request, res: Trigger.Response) -> Trigger.Response:
        if not self._run_lock.acquire(blocking=False):
            res.success = False
            res.message = json.dumps({"error": "busy"}, separators=(",", ":"))
            return res

        try:
            listen_s = float(self.get_parameter("listen_duration_s").value)
            if listen_s <= 0:
                listen_s = 1.0

            call_rust = bool(self.get_parameter("call_rust_service").value)
            call_pcb = bool(self.get_parameter("call_pcb_infer_service").value)
            rust_to = float(self.get_parameter("rust_service_timeout_s").value)
            pcb_to = float(self.get_parameter("pcb_service_timeout_s").value)

            window_start = _now_mono()

            rust_resp = None
            pcb_resp = None
            rust_err = ""
            pcb_err = ""

            if call_pcb:
                ok, resp, err = self._call_trigger(self._pcb_cli, pcb_to)
                pcb_resp, pcb_err = resp, err if not ok else err
            if call_rust:
                ok, resp, err = self._call_trigger(self._rust_cli, rust_to)
                rust_resp, rust_err = resp, err if not ok else err

            end_t = window_start + listen_s
            while _now_mono() < end_t:
                time.sleep(0.02)

            snap = self._snapshot_from_live(window_start)

            rust_service_dict = None
            rust_detected = None

            if rust_resp is not None:
                rust_detected = bool(getattr(rust_resp, "success", False))
                msg = str(getattr(rust_resp, "message", ""))
                d = _safe_json_loads(msg)
                if isinstance(d, dict):
                    rust_service_dict = d
                    if "detected" in d:
                        rust_detected = bool(d["detected"])
                else:
                    rust_service_dict = {"raw_message": msg}
            else:
                rust_service_dict = None

            if rust_detected is None:
                rust_detected = snap.get("rust_det_from_topic", False)

            pcb_report_dict = snap.get("pcb_report_dict", None)
            pcb_ok = snap.get("pcb_ok", None)

            with self._snap_lock:
                if snap["rust_prob_msg"] is not None:
                    self._snap_rust_prob = snap["rust_prob_msg"]
                if snap["pcb_annot_msg"] is not None:
                    self._snap_pcb_annot = snap["pcb_annot_msg"]

                self._snap_rust_detected = bool(rust_detected) if rust_detected is not None else None
                self._snap_pcb_ok = bool(pcb_ok) if pcb_ok is not None else None

                self._last_rust_service = rust_service_dict
                self._last_pcb_report = pcb_report_dict

                meta = {
                    "listen_duration_s": listen_s,
                    "rust_service_called": call_rust,
                    "pcb_service_called": call_pcb,
                    "rust_service_error": rust_err,
                    "pcb_service_error": pcb_err,
                    "got_rust_prob": bool(snap["got_rust_prob"]),
                    "got_pcb_annot": bool(snap["got_pcb_annot"]),
                    "got_pcb_report": bool(snap["got_pcb_report"]),
                }

                combined = self._compose_combined_report(
                    rust_service_dict=rust_service_dict,
                    pcb_report_dict=pcb_report_dict,
                    rust_detected=self._snap_rust_detected,
                    pcb_ok=self._snap_pcb_ok,
                    meta=meta,
                )
                self._snap_report_str = combined
                self._last_session_summary = _safe_json_loads(combined)

            self._publish_snapshot_once()

            have_any = bool(snap["got_pcb_report"] or (rust_resp is not None))
            res.success = bool(have_any)
            res.message = self._snap_report_str
            return res

        finally:
            self._run_lock.release()

    # ---------------- republish loop ----------------

    def _publish_snapshot_once(self) -> None:
        with self._snap_lock:
            rust_prob = self._snap_rust_prob
            pcb_annot = self._snap_pcb_annot
            rust_det = self._snap_rust_detected
            pcb_ok = self._snap_pcb_ok
            rep = self._snap_report_str

        if rust_prob is not None:
            self._pub_rust_prob.publish(rust_prob)
        if pcb_annot is not None:
            self._pub_pcb_annot.publish(pcb_annot)

        if rust_det is not None:
            m = Bool()
            m.data = bool(rust_det)
            self._pub_rust_det.publish(m)

        if pcb_ok is not None:
            m = Bool()
            m.data = bool(pcb_ok)
            self._pub_pcb_ok.publish(m)

        if rep:
            s = String()
            s.data = rep
            self._pub_report.publish(s)

    def _on_republish_timer(self) -> None:
        self._publish_snapshot_once()


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
