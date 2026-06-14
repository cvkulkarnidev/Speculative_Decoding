from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from .assistant import assistant_rollout, compute_rollout_loss_and_metrics, load_assistant_model, shift_left
from .config import Eagle3TrainingConfig
from .data import SupervisedDataCollator, SupervisedJsonlDataset


def load_checkpoint(cfg: Eagle3TrainingConfig, checkpoint_path: str, dtype: torch.dtype) -> torch.nn.Module:
    path = Path(checkpoint_path)
    if (path / "adapter_config.json").exists():
        adapter_model = load_assistant_model(
            cfg.assistant_model_path,
            dtype=dtype,
            trust_remote_code=cfg.trust_remote_code,
            adapter_path=str(path),
            is_trainable=False,
        )
        return adapter_model.merge_and_unload()
    return load_assistant_model(
        str(path),
        dtype=dtype,
        trust_remote_code=cfg.trust_remote_code,
        is_trainable=False,
    )


@torch.no_grad()
def run_eval(cfg: Eagle3TrainingConfig, checkpoint_path: str, data_dir: str) -> dict[str, float]:
    accelerator = Accelerator(mixed_precision="bf16" if cfg.bf16 else "fp16" if cfg.fp16 else "no")
    dtype = torch.bfloat16 if cfg.bf16 else torch.float16 if cfg.fp16 else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(cfg.target_model_path, trust_remote_code=cfg.trust_remote_code)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    target_model = AutoModelForCausalLM.from_pretrained(
        cfg.target_model_path,
        dtype=dtype,
        trust_remote_code=cfg.trust_remote_code,
    )
    target_model.eval().requires_grad_(False)
    assistant_model = load_checkpoint(cfg, checkpoint_path, dtype)
    assistant_model.eval()
    dataloader = DataLoader(
        SupervisedJsonlDataset(data_dir),
        batch_size=cfg.per_device_eval_batch_size,
        shuffle=False,
        collate_fn=SupervisedDataCollator(tokenizer.pad_token_id, cfg.max_length),
    )
    target_model, assistant_model, dataloader = accelerator.prepare(target_model, assistant_model, dataloader)

    losses: list[torch.Tensor] = []
    correct = torch.tensor(0.0, device=accelerator.device)
    total = torch.tensor(0.0, device=accelerator.device)
    for batch in tqdm(dataloader, desc="Evaluating", disable=not accelerator.is_main_process):
        assistant_logits, target_logits = assistant_rollout(
            target_model,
            assistant_model,
            batch["input_ids"],
            batch["attention_mask"],
            return_target_logits=cfg.kl_weight > 0,
            steps=cfg.rollout_steps,
            pad_token_id=tokenizer.pad_token_id,
        )
        loss, _ = compute_rollout_loss_and_metrics(
            assistant_logits,
            target_logits,
            batch["labels"],
            temperature=cfg.temperature,
            kl_weight=cfg.kl_weight,
            ce_weight=cfg.ce_weight,
            decay=cfg.rollout_decay,
        )
        labels = shift_left(batch["labels"], -100)[:, 1:]
        valid = labels != -100
        predictions = assistant_logits[0][:, :-1].argmax(dim=-1)
        losses.append(accelerator.gather_for_metrics(loss.detach()).mean())
        correct += accelerator.gather_for_metrics(((predictions == labels) & valid).sum()).sum()
        total += accelerator.gather_for_metrics(valid.sum()).sum()
    mean_loss = torch.stack(losses).mean().item() if losses else 0.0
    return {
        "loss": mean_loss,
        "accuracy": (correct / total).item() if total.item() else 0.0,
        "perplexity": math.exp(min(mean_loss, 20.0)),
        "target_tokens": float(total.item()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--data_dir", default=None)
    args = parser.parse_args()
    cfg = Eagle3TrainingConfig.from_yaml(args.config)
    data_dir = args.data_dir or cfg.eval_data_dir or cfg.test_data_dir
    if not data_dir:
        raise ValueError("Pass --data_dir or configure eval_data_dir/test_data_dir.")
    print(json.dumps(run_eval(cfg, args.checkpoint_path, data_dir), indent=2))


if __name__ == "__main__":
    main()
