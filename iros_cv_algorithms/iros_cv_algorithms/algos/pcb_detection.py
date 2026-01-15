# iros_cv_algorithms/algos/rust_detection.py

from __future__ import annotations

from typing import Any, Dict
from .interface import CvAlgorithm

class PCBDetectionAlgorithm(CvAlgorithm):
    @property
    def key(self) -> str:
        return "pcb_detection"

    def run(self, image_bgr) -> Dict[str, Any]:
        # Заглушка:
        return {"status": "todo", "info": "implement using existing rust detection logic"}
