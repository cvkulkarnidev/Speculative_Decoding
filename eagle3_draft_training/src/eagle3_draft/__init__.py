"""Gemma 4 assistant checkpoint fine-tuning package."""

from typing import Any

__all__ = ["Eagle3TrainingConfig"]


def __getattr__(name: str) -> Any:
    if name == "Eagle3TrainingConfig":
        from .config import Eagle3TrainingConfig

        return Eagle3TrainingConfig
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
