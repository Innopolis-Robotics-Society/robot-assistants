# iros_cv_algorithms/algos/interface.py

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class CvAlgorithm(ABC):
    """Единый интерфейс для CV-алгоритмов."""

    @property
    @abstractmethod
    def key(self) -> str:
        """
        Уникальный ключ алгоритма.
        Нода будет публиковать в топик: <result_prefix>/<key>
        """
        raise NotImplementedError

    @abstractmethod
    def run(self, image_bgr) -> Any:
        """
        На вход: OpenCV BGR image (numpy array).
        На выход: результат (желательно JSON-совместимый: dict/list/str/float/int).
        """
        raise NotImplementedError
