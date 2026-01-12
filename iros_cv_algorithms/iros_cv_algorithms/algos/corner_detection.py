# iros_cv_algorithms/algos/corner_detection.py

from __future__ import annotations

from typing import Any, Dict
from .interface import CvAlgorithm

class CornerDetectionAlgorithm(CvAlgorithm):
    @property
    def key(self) -> str:
        return "corner_detection"

    def run(self, image_bgr) -> Dict[str, Any]:
        # Заглушка:
        return {"status": "todo", "info": "implement using existing corner detection logic"}
