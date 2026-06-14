from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass
class Eagle3TrainingConfig:
    train_data_dir: str
    output_dir: str
    assistant_model_path: str
    target_model_path: str

    eval_data_dir: str | None = None
    test_data_dir: str | None = None
    assistant_adapter_path: str | None = None
    tensorboard_log_dir: str | None = None

    max_length: int = 2048
    learning_rate: float = 2e-5
    weight_decay: float = 0.0
    warmup_steps: int = 100
    gradient_accumulation_steps: int = 4
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    num_train_epochs: int = 1
    max_steps: int = -1
    logging_steps: int = 10
    eval_steps: int = 200
    save_steps: int = 500
    seed: int = 42

    temperature: float = 1.0
    kl_weight: float = 0.0
    ce_weight: float = 1.0
    rollout_steps: int = 1
    rollout_decay: float = 0.8
    bf16: bool = True
    fp16: bool = False
    gradient_checkpointing: bool = True
    trust_remote_code: bool = True

    use_lora: bool = False
    merge_lora_on_save: bool = True
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Eagle3TrainingConfig":
        with open(path, "r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
        allowed = {field.name for field in fields(cls)}
        unknown = set(raw) - allowed
        if unknown:
            raise ValueError(f"Unknown config keys: {sorted(unknown)}")
        cfg = cls(**raw)
        if cfg.bf16 and cfg.fp16:
            raise ValueError("Only one of bf16 or fp16 may be enabled.")
        if cfg.kl_weight < 0 or cfg.ce_weight < 0 or cfg.kl_weight + cfg.ce_weight <= 0:
            raise ValueError("kl_weight and ce_weight must be non-negative with a positive sum.")
        if cfg.rollout_steps < 1:
            raise ValueError("rollout_steps must be at least 1.")
        if not 0 < cfg.rollout_decay <= 1:
            raise ValueError("rollout_decay must be in (0, 1].")
        return cfg

    @property
    def resolved_target_model_path(self) -> str:
        return self.target_model_path

    @property
    def resolved_assistant_model_path(self) -> str:
        return self.assistant_model_path

    def to_dict(self) -> dict[str, Any]:
        return {field.name: getattr(self, field.name) for field in fields(self)}
