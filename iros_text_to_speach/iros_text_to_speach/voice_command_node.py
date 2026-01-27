#!/usr/bin/env python3
from __future__ import annotations

import json
import threading
import time
import queue
from typing import Any, Dict, Optional, List, Tuple

import numpy as np
import sounddevice as sd

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from std_msgs.msg import String

from kokoro import KPipeline


class KokoroTTS:
    def __init__(self, voice: str = "af_bella"):
        self.voice = voice
        self.pipeline: Optional[KPipeline] = None
        self.sample_rate = 24000

    def initialize(self, logger) -> bool:
        logger.info("Loading Kokoro TTS model...")
        self.pipeline = KPipeline(lang_code="a")  # auto language
        _ = self.synthesize("System ready.")
        logger.info("Kokoro TTS ready.")
        return True

    def synthesize(self, text: str) -> Tuple[np.ndarray, int]:
        if self.pipeline is None:
            raise RuntimeError("Kokoro pipeline is not initialized")

        audio = None
        for _, _, audio_chunk in self.pipeline(text, voice=self.voice):
            audio = audio_chunk

        if audio is None:
            raise RuntimeError("Kokoro produced no audio")

        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        return audio, self.sample_rate


class CvHubReportVoiceNode(Node):
    """
    No anti-spam:
      - Every incoming /cv_hub/report triggers speech immediately.
      - While audio is playing, new reports are ignored.

    OK rule:
      rust.detected == False AND pcb.ok == True AND gear.ok == True
    """

    def __init__(self):
        super().__init__("cv_hub_report_voice_node")
        self._cbg = ReentrantCallbackGroup()

        # params
        self.declare_parameter("report_topic", "/cv_hub/report")
        self.declare_parameter("voice", "af_bella")
        self.declare_parameter("class_name_map_json", json.dumps({
            "blue-board": "blue board",
            "digit-board": "digit board",
            "esp": "ESP module",
            "green-board": "green board",
            "purple-board": "purple board",
        }, ensure_ascii=False))

        report_topic = str(self.get_parameter("report_topic").value)
        voice = str(self.get_parameter("voice").value)

        # TTS
        self.tts = KokoroTTS(voice=voice)
        self._tts_ready = threading.Event()

        # playback state (single utterance at a time)
        self.is_playing = False
        self._play_lock = threading.Lock()

        # init TTS in background
        threading.Thread(target=self._init_tts, daemon=True).start()

        # subscribe
        self.subscription = self.create_subscription(
            String,
            report_topic,
            self._on_report,
            10,
            callback_group=self._cbg,
        )

        self.get_logger().info(f"Listening to: {report_topic}")

    # -------- TTS init / playback --------

    def _init_tts(self):
        try:
            self.tts.initialize(self.get_logger())
            self._tts_ready.set()
        except Exception as e:
            self.get_logger().error(f"Failed to initialize Kokoro TTS: {e}")

    def _speak_blocking(self, text: str):
        try:
            if not self._tts_ready.wait(timeout=30.0):
                self.get_logger().error("TTS not ready (timeout).")
                return

            audio, sr = self.tts.synthesize(text)
            self.get_logger().info(f"Playing: {text}")
            sd.play(audio, sr)
            sd.wait()

        except Exception as e:
            self.get_logger().error(f"Speech failed: {e}")

        finally:
            with self._play_lock:
                self.is_playing = False

    # -------- report parsing / phrasing --------

    @staticmethod
    def _safe_json_loads(s: str) -> Optional[dict]:
        try:
            return json.loads(s)
        except Exception:
            return None

    @staticmethod
    def _as_bool(v: Any) -> Optional[bool]:
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
        return None

    @staticmethod
    def _int_or_zero(v: Any) -> int:
        try:
            return int(v)
        except Exception:
            return 0

    def _get_class_name_map(self) -> Dict[str, str]:
        raw = str(self.get_parameter("class_name_map_json").value)
        d = self._safe_json_loads(raw)
        return d if isinstance(d, dict) else {}

    def _compute_missing_components(self, pcb_report_data: dict) -> List[Tuple[str, int]]:
        counts_ref = pcb_report_data.get("counts_ref", {})
        counts_cur = pcb_report_data.get("counts_cur", {})
        if not isinstance(counts_ref, dict) or not isinstance(counts_cur, dict):
            return []

        missing: List[Tuple[str, int]] = []
        for k, ref_v in counts_ref.items():
            ref_n = self._int_or_zero(ref_v)
            cur_n = self._int_or_zero(counts_cur.get(k, 0))
            if cur_n < ref_n:
                missing.append((str(k), ref_n - cur_n))
        return missing

    def _compose_phrase(self, report: dict) -> str:
        rust_detected = self._as_bool(((report.get("rust") or {}).get("detected")))
        pcb_ok = self._as_bool(((report.get("pcb") or {}).get("ok")))
        gear_ok = self._as_bool(((report.get("gear") or {}).get("ok")))

        overall_ok = (rust_detected is False) and (pcb_ok is True) and (gear_ok is True)

        if overall_ok:
            return "All components are present. No rust detected. Gears are OK."

        issues: List[str] = []

        # rust
        if rust_detected is True:
            issues.append("rust detected")
        elif rust_detected is None:
            issues.append("no rust result available")

        # pcb
        if pcb_ok is not True:
            pcb_block = report.get("pcb") or {}
            pcb_report_data = pcb_block.get("report_data") or {}
            missing = []
            if isinstance(pcb_report_data, dict):
                missing = self._compute_missing_components(pcb_report_data)

            if missing:
                name_map = self._get_class_name_map()
                parts = []
                for cls, n in missing:
                    spoken = name_map.get(cls, cls)
                    parts.append(spoken if n == 1 else f"{spoken} x{n}")
                issues.append("missing components: " + ", ".join(parts))
            else:
                reason = None
                if isinstance(pcb_report_data, dict):
                    reason = pcb_report_data.get("reason")
                if not reason:
                    pcb_report = pcb_block.get("report") or {}
                    if isinstance(pcb_report, dict):
                        reason = (pcb_report.get("service") or {}).get("error")
                issues.append(f"PCB check failed ({reason or 'unknown reason'})")

        # gear
        if gear_ok is not True:
            if gear_ok is None:
                issues.append("no gear result available")
            else:
                issues.append("gear check failed")

        return "Problems detected: " + "; ".join(issues) + "."

    # -------- ROS callback --------

    def _on_report(self, msg: String):
        # ignore while speaking
        with self._play_lock:
            if self.is_playing:
                return
            # reserve playback slot immediately
            self.is_playing = True

        report = self._safe_json_loads(msg.data)
        if not isinstance(report, dict):
            self.get_logger().warning("Invalid /cv_hub/report JSON.")
            with self._play_lock:
                self.is_playing = False
            return

        text = self._compose_phrase(report)

        # speak in background (blocking playback)
        threading.Thread(target=self._speak_blocking, args=(text,), daemon=True).start()


def main(args=None):
    rclpy.init(args=args)
    node = CvHubReportVoiceNode()

    ex = MultiThreadedExecutor(num_threads=4)
    ex.add_node(node)
    try:
        ex.spin()
    except KeyboardInterrupt:
        pass
    finally:
        ex.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
