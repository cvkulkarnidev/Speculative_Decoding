from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

from .compatibility import validate_assistant_config


def unwrap_module(model: torch.nn.Module) -> torch.nn.Module:
    while hasattr(model, "module"):
        model = model.module
    return model


def load_assistant_model(
    model_path: str,
    *,
    dtype: torch.dtype,
    trust_remote_code: bool,
    adapter_path: str | None = None,
    is_trainable: bool = True,
) -> torch.nn.Module:
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=dtype,
        trust_remote_code=trust_remote_code,
    )
    validate_assistant_config(model.config)
    if adapter_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=is_trainable)
    return model


def add_lora(
    model: torch.nn.Module,
    *,
    rank: int,
    alpha: int,
    dropout: float,
) -> torch.nn.Module:
    from peft import LoraConfig, get_peft_model

    return get_peft_model(
        model,
        LoraConfig(
            r=rank,
            lora_alpha=alpha,
            lora_dropout=dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules="all-linear",
        ),
    )


def shift_left(tensor: torch.Tensor, fill_value: int | float) -> torch.Tensor:
    fill = torch.full_like(tensor[:, :1], fill_value)
    return torch.cat([tensor[:, 1:], fill], dim=1)


def assistant_rollout(
    target_model: torch.nn.Module,
    assistant_model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    return_target_logits: bool,
    steps: int,
    pad_token_id: int,
) -> tuple[list[torch.Tensor], torch.Tensor | None]:
    with torch.no_grad():
        target_base = unwrap_module(target_model)
        target_outputs = target_base(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_shared_kv_states=True,
            use_cache=False,
            logits_to_keep=0 if return_target_logits else 1,
        )
        if target_outputs.hidden_states is None or target_outputs.shared_kv_states is None:
            raise ValueError("Target model did not return hidden_states and shared_kv_states.")
        target_logits = target_outputs.logits if return_target_logits else None

    position_ids = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
    position_ids = position_ids.expand(input_ids.shape[0], -1)
    current_ids = input_ids
    current_hidden = target_outputs.hidden_states[-1]
    logits_per_step: list[torch.Tensor] = []
    for _ in range(steps):
        token_embeddings = target_base.get_input_embeddings()(current_ids)
        assistant_inputs = torch.cat([token_embeddings, current_hidden], dim=-1)
        assistant_outputs = assistant_model(
            inputs_embeds=assistant_inputs,
            attention_mask=attention_mask,
            position_ids=position_ids,
            shared_kv_states=target_outputs.shared_kv_states,
            use_cache=False,
        )
        logits_per_step.append(assistant_outputs.logits)
        current_hidden = assistant_outputs.last_hidden_state
        current_ids = shift_left(current_ids, pad_token_id)
    return logits_per_step, target_logits


def compute_loss_and_metrics(
    assistant_logits: torch.Tensor,
    target_logits: torch.Tensor | None,
    labels: torch.Tensor,
    *,
    temperature: float,
    kl_weight: float,
    ce_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    draft = assistant_logits[:, :-1].float()
    shifted_labels = labels[:, 1:]
    valid = shifted_labels != -100
    if not valid.any():
        zero = draft.sum() * 0.0
        return zero, {"loss": 0.0, "ce_loss": 0.0, "kl_loss": 0.0, "accuracy": 0.0}

    draft_active = draft[valid]
    label_active = shifted_labels[valid]
    ce_loss = F.cross_entropy(draft_active, label_active)
    if kl_weight > 0:
        if target_logits is None:
            raise ValueError("target_logits are required when kl_weight is positive.")
        target_active = target_logits[:, :-1].float()[valid]
        temp = float(temperature)
        kl_loss = F.kl_div(
            F.log_softmax(draft_active / temp, dim=-1),
            F.softmax(target_active / temp, dim=-1),
            reduction="batchmean",
        ) * (temp * temp)
    else:
        kl_loss = ce_loss.detach() * 0.0
    loss = ce_weight * ce_loss + kl_weight * kl_loss
    accuracy = (draft_active.argmax(dim=-1) == label_active).float().mean().item()
    return loss, {
        "loss": float(loss.detach().item()),
        "ce_loss": float(ce_loss.detach().item()),
        "kl_loss": float(kl_loss.detach().item()),
        "accuracy": accuracy,
    }


def compute_rollout_loss_and_metrics(
    assistant_logits: list[torch.Tensor],
    target_logits: torch.Tensor | None,
    labels: torch.Tensor,
    *,
    temperature: float,
    kl_weight: float,
    ce_weight: float,
    decay: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    if not assistant_logits:
        raise ValueError("assistant_logits must contain at least one rollout step.")
    weighted_losses: list[torch.Tensor] = []
    metrics: dict[str, float] = {}
    current_labels = labels
    current_target_logits = target_logits
    total_weight = 0.0
    for step, logits in enumerate(assistant_logits):
        weight = decay**step
        step_loss, step_metrics = compute_loss_and_metrics(
            logits,
            current_target_logits,
            current_labels,
            temperature=temperature,
            kl_weight=kl_weight,
            ce_weight=ce_weight,
        )
        weighted_losses.append(step_loss * weight)
        total_weight += weight
        metrics[f"step_{step + 1}_loss"] = step_metrics["loss"]
        metrics[f"step_{step + 1}_accuracy"] = step_metrics["accuracy"]
        current_labels = shift_left(current_labels, -100)
        if current_target_logits is not None:
            current_target_logits = shift_left(current_target_logits, 0.0)
    loss = sum(weighted_losses) / total_weight
    metrics["loss"] = float(loss.detach().item())
    metrics["accuracy"] = metrics["step_1_accuracy"]
    return loss, metrics


def save_assistant_checkpoint(
    model: torch.nn.Module,
    tokenizer: Any,
    output_dir: str | Path,
    training_metadata: dict[str, Any] | None = None,
) -> None:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    if training_metadata is not None:
        (path / "training_metadata.json").write_text(
            json.dumps(training_metadata, indent=2),
            encoding="utf-8",
        )
