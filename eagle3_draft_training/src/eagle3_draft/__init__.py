"""Lightweight EAGLE-3-style drafter training package."""

from typing import Any

__all__ = ["Eagle3TrainingConfig", "Eagle3DraftModel"]


def __getattr__(name: str) -> Any:
    if name == "Eagle3TrainingConfig":
        from .config import Eagle3TrainingConfig

        return Eagle3TrainingConfig
    if name == "Eagle3DraftModel":
        from .model import Eagle3DraftModel

        return Eagle3DraftModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
