from __future__ import annotations

from typing import Any


SUPPORTED_ASSISTANT_MODEL_TYPES = {"gemma4_assistant", "gemma4_unified_assistant"}


def validate_assistant_config(config: Any) -> None:
    model_type = getattr(config, "model_type", None)
    if model_type not in SUPPORTED_ASSISTANT_MODEL_TYPES:
        raise ValueError(
            "Expected a Gemma 4 *-assistant checkpoint with model_type "
            f"{sorted(SUPPORTED_ASSISTANT_MODEL_TYPES)}, got {model_type!r}."
        )
