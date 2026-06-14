from __future__ import annotations

from contextlib import contextmanager
from importlib import import_module
from typing import Any, Iterator


@contextmanager
def disable_bitsandbytes_dispatch(lora_model_module: Any | None = None) -> Iterator[None]:
    """Prevent PEFT from importing an unused, broken bitsandbytes installation."""
    module = lora_model_module or import_module("peft.tuners.lora.model")
    original_bnb = module.is_bnb_available
    original_bnb_4bit = module.is_bnb_4bit_available
    module.is_bnb_available = lambda: False
    module.is_bnb_4bit_available = lambda: False
    try:
        yield
    finally:
        module.is_bnb_available = original_bnb
        module.is_bnb_4bit_available = original_bnb_4bit
