from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from data_utils import DatasetBundle
from model_utils import ModelBundle


class BaseMethod(ABC):
    """Interface for all contamination detection methods."""

    name: str = ""

    @abstractmethod
    def run(self, model_bundle: ModelBundle, dataset: DatasetBundle) -> dict[str, Any]:
        """Run this method on one dataset and return a JSON-serializable result."""
        raise NotImplementedError
