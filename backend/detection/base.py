"""
Abstract base class for anomaly detection strategies.

All concrete strategies must implement `detect`, which receives a plain
dictionary of features and returns a StrategyResult.
"""

from abc import ABC, abstractmethod
from typing import Any

from backend.models.schemas import StrategyResult


class BaseDetector(ABC):
    """Common interface shared by every detection strategy."""

    @abstractmethod
    async def detect(self, data: dict[str, Any]) -> StrategyResult:
        """
        Analyse the provided data record and return a detection result.

        Args:
            data: Flat or nested dictionary of feature values.

        Returns:
            StrategyResult indicating whether an anomaly was detected.
        """
        ...

    async def health_check(self) -> bool:
        """Return True if the underlying model/service is reachable."""
        ...
