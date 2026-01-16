# iros_cv_algorithms/algos/pcb_detection.py

from __future__ import annotations

import math
import threading
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from .interface import CvAlgorithm


@dataclass(frozen=True)
class _Feat:
    cls: str
    x: float  # center x in [0..1]
    y: float  # center y in [0..1]
    w: float  # width  in [0..1]
    h: float  # height in [0..1]
    conf: float


def _rel_diff(a: float, b: float, eps: float = 1e-9) -> float:
    return abs(a - b) / max(abs(a), eps)


class PCBDetectionAlgorithm(CvAlgorithm):
    """
    Calibration on first use:
      - collect N signatures from N frames
      - choose baseline signature
      - compute per-object tolerances as max observed deviations (+ margins), with minimal floors

    Runtime:
      - strict counts per class
      - per-object geometry check (dx/dy + relative dw/dh)
      - distances-to-anchors (simple global check)

    Additionally returns `detections` (YOLO normalized boxes) so the node can draw/save overlays.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        calib_samples: int = 5,
        conf_thr: float = 0.25,
        iou_thr: float = 0.5,
        device: str = "0",
        # tolerance floors
        min_pos_tol: float = 0.02,
        min_size_tol: float = 0.25,
        min_dist_tol: float = 0.04,
        # margins added to observed maxima
        pos_margin: float = 0.01,
        size_margin: float = 0.05,
        dist_margin: float = 0.01,
    ):
        self._conf_thr = float(conf_thr)
        self._iou_thr = float(iou_thr)
        self._device = device

        self._calib_need = max(1, int(calib_samples))
        self._min_pos_tol = float(min_pos_tol)
        self._min_size_tol = float(min_size_tol)
        self._min_dist_tol = float(min_dist_tol)
        self._pos_margin = float(pos_margin)
        self._size_margin = float(size_margin)
        self._dist_margin = float(dist_margin)

        self._lock = threading.Lock()
        self._calib_buf: List[Dict[str, List[_Feat]]] = []

        # baseline state
        self._baseline_sig: Optional[Dict[str, List[_Feat]]] = None
        self._baseline_counts: Optional[Dict[str, int]] = None
        self._tol_geom: Optional[Dict[str, List[Dict[str, float]]]] = None  # per class, per index
        self._anchors: Optional[List[int]] = None
        self._tol_dist: Optional[float] = None

        from ultralytics import YOLO  # dependency

        if model_path is None:
            model_path = str(Path(__file__).resolve().parent / "models" / "yolo12s-pcb.pt")

        self._model_path = model_path
        self._model = YOLO(self._model_path)

    @property
    def key(self) -> str:
        return "pcb_detection"

    def run(self, image_bgr) -> Dict[str, Any]:
        feats = self._infer(image_bgr)
        sig = self._build_signature(feats)
        dets_out = [asdict(f) for f in feats]  # for node overlays

        with self._lock:
            # already calibrated
            if self._baseline_sig is not None:
                out = self._check(sig)
                out["detections"] = dets_out
                return out

            # calibrating
            self._calib_buf.append(sig)
            have = len(self._calib_buf)
            need = self._calib_need

            if have < need:
                return {
                    "baseline_set": False,
                    "calibrating": True,
                    "progress": {"have": have, "need": need},
                    "counts_now": {k: len(v) for k, v in sig.items()},
                    "detections": dets_out,
                }

            # build baseline now
            ok = self._build_baseline(self._calib_buf)
            self._calib_buf.clear()

            if not ok:
                return {
                    "baseline_set": False,
                    "calibrating": False,
                    "ok": False,
                    "reason": "baseline_build_failed_retry",
                    "detections": dets_out,
                }

            return {
                "baseline_set": True,
                "calibrating": False,
                "ok": True,
                "reason": "baseline_created",
                "counts_ref": self._baseline_counts,
                "dist_tol": self._tol_dist,
                "anchors": self._anchors,
                "geom_tol_summary": self._tol_geom,  # can be big, but useful for debugging
                "detections": dets_out,
            }

    # ---------------- baseline build / runtime checks ----------------

    def _build_baseline(self, sigs: List[Dict[str, List[_Feat]]]) -> bool:
        def total(s: Dict[str, List[_Feat]]) -> int:
            return sum(len(v) for v in s.values())

        base = max(sigs, key=total)
        base_counts = {k: len(v) for k, v in base.items()}

        # keep only samples that match baseline counts exactly
        good = []
        for s in sigs:
            c = {k: len(v) for k, v in s.items()}
            if c == base_counts:
                good.append(s)

        if len(good) < 2:
            return False

        # per-object tolerances
        tol_geom: Dict[str, List[Dict[str, float]]] = {}
        for cls, base_list in base.items():
            tol_geom[cls] = []
            for i, fb in enumerate(base_list):
                max_dx = 0.0
                max_dy = 0.0
                max_dw = 0.0
                max_dh = 0.0

                for s in good:
                    f = s[cls][i]
                    max_dx = max(max_dx, abs(f.x - fb.x))
                    max_dy = max(max_dy, abs(f.y - fb.y))
                    max_dw = max(max_dw, _rel_diff(fb.w, f.w))
                    max_dh = max(max_dh, _rel_diff(fb.h, f.h))

                tol_geom[cls].append({
                    "dx": max(self._min_pos_tol, max_dx + self._pos_margin),
                    "dy": max(self._min_pos_tol, max_dy + self._pos_margin),
                    "dw": max(self._min_size_tol, max_dw + self._size_margin),
                    "dh": max(self._min_size_tol, max_dh + self._size_margin),
                })

        # anchors + global distance tolerance
        base_all = self._flatten_sorted(base)
        n = len(base_all)
        if n >= 2:
            anchors = list(dict.fromkeys([0, n - 1, n // 2]))
            areas = [f.w * f.h for f in base_all]
            anchors.append(int(max(range(n), key=lambda i: areas[i])))
            anchors = list(dict.fromkeys(anchors))
        else:
            anchors = [0]

        max_dd = 0.0
        for s in good:
            cur_all = self._flatten_sorted(s)
            if len(cur_all) != len(base_all):
                continue
            for i in range(len(base_all)):
                for aidx in anchors:
                    db = math.hypot(base_all[i].x - base_all[aidx].x, base_all[i].y - base_all[aidx].y)
                    dc = math.hypot(cur_all[i].x - cur_all[aidx].x, cur_all[i].y - cur_all[aidx].y)
                    max_dd = max(max_dd, abs(db - dc))

        tol_dist = max(self._min_dist_tol, max_dd + self._dist_margin)

        self._baseline_sig = base
        self._baseline_counts = base_counts
        self._tol_geom = tol_geom
        self._anchors = anchors
        self._tol_dist = tol_dist
        return True

    def _check(self, cur: Dict[str, List[_Feat]]) -> Dict[str, Any]:
        assert self._baseline_sig is not None
        assert self._baseline_counts is not None
        assert self._tol_geom is not None
        assert self._anchors is not None
        assert self._tol_dist is not None

        # counts strict
        cur_counts = {k: len(v) for k, v in cur.items()}
        if cur_counts != self._baseline_counts:
            return {
                "baseline_set": True,
                "ok": False,
                "reason": "counts_mismatch",
                "counts_ref": self._baseline_counts,
                "counts_cur": cur_counts,
            }

        # geometry
        mism = []
        for cls, base_list in self._baseline_sig.items():
            cur_list = cur[cls]
            for i, (fb, fc) in enumerate(zip(base_list, cur_list)):
                tol = self._tol_geom[cls][i]
                dx = abs(fc.x - fb.x)
                dy = abs(fc.y - fb.y)
                dw = _rel_diff(fb.w, fc.w)
                dh = _rel_diff(fb.h, fc.h)
                if dx > tol["dx"] or dy > tol["dy"] or dw > tol["dw"] or dh > tol["dh"]:
                    mism.append({
                        "class": cls,
                        "index": i,
                        "dx": dx, "dy": dy, "dw_rel": dw, "dh_rel": dh,
                        "tol": tol,
                    })

        if mism:
            return {
                "baseline_set": True,
                "ok": False,
                "reason": "geometry_mismatch",
                "mismatches": mism[:50],
            }

        # distances
        base_all = self._flatten_sorted(self._baseline_sig)
        cur_all = self._flatten_sorted(cur)
        bad = []
        for i in range(len(base_all)):
            for aidx in self._anchors:
                db = math.hypot(base_all[i].x - base_all[aidx].x, base_all[i].y - base_all[aidx].y)
                dc = math.hypot(cur_all[i].x - cur_all[aidx].x, cur_all[i].y - cur_all[aidx].y)
                if abs(db - dc) > self._tol_dist:
                    bad.append({"i": i, "a": aidx, "abs_diff": abs(db - dc)})

        if bad:
            return {
                "baseline_set": True,
                "ok": False,
                "reason": "distance_mismatch",
                "dist_tol": self._tol_dist,
                "mismatches": bad[:50],
            }

        return {"baseline_set": True, "ok": True, "reason": "ok"}

    # ---------------- inference + signature helpers ----------------

    def _infer(self, image_bgr) -> List[_Feat]:
        results = self._model.predict(
            source=image_bgr,
            conf=self._conf_thr,
            iou=self._iou_thr,
            device=self._device,
            verbose=False,
        )
        r = results[0]
        if r.boxes is None or len(r.boxes) == 0:
            return []

        names = self._model.names
        xywhn = r.boxes.xywhn.cpu().numpy()
        cls = r.boxes.cls.cpu().numpy().astype(int)
        conf = r.boxes.conf.cpu().numpy()

        feats: List[_Feat] = []
        for (x, y, w, h), c, p in zip(xywhn, cls, conf):
            cls_name = str(names.get(int(c), str(int(c))))
            feats.append(_Feat(cls=cls_name, x=float(x), y=float(y), w=float(w), h=float(h), conf=float(p)))
        return feats

    def _build_signature(self, feats: List[_Feat]) -> Dict[str, List[_Feat]]:
        sig: Dict[str, List[_Feat]] = {}
        for f in feats:
            sig.setdefault(f.cls, []).append(f)
        for k in sig:
            sig[k].sort(key=lambda z: (z.x, z.y))
        return sig

    def _flatten_sorted(self, sig: Dict[str, List[_Feat]]) -> List[_Feat]:
        all_feats: List[_Feat] = []
        for k in sorted(sig.keys()):
            all_feats.extend(sig[k])
        all_feats.sort(key=lambda z: (z.x, z.y, z.cls))
        return all_feats
