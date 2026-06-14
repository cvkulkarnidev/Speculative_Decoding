from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

from .assistant import (
    add_lora,
    assistant_rollout,
    compute_rollout_loss_and_metrics,
    load_assistant_model,
    save_assistant_checkpoint,
    shift_left,
)
from .config import Eagle3TrainingConfig
from .data import SupervisedDataCollator, SupervisedJsonlDataset


def make_loader(
    data_dir: str | None,
    tokenizer: AutoTokenizer,
    cfg: Eagle3TrainingConfig,
    *,
    train: bool,
) -> DataLoader | None:
    if not data_dir or not (Path(data_dir) / "data.pt").exists():
        return None
    return DataLoader(
        SupervisedJsonlDataset(data_dir),
        batch_size=cfg.per_device_train_batch_size if train else cfg.per_device_eval_batch_size,
        shuffle=train,
        collate_fn=SupervisedDataCollator(tokenizer.pad_token_id, cfg.max_length),
    )


@torch.no_grad()
def evaluate(
    accelerator: Accelerator,
    target_model: torch.nn.Module,
    assistant_model: torch.nn.Module,
    dataloader: DataLoader | None,
    cfg: Eagle3TrainingConfig,
    pad_token_id: int,
) -> dict[str, float]:
    if dataloader is None:
        return {}
    assistant_model.eval()
    losses: list[torch.Tensor] = []
    correct = torch.tensor(0.0, device=accelerator.device)
    total = torch.tensor(0.0, device=accelerator.device)
    for batch in dataloader:
        assistant_logits, target_logits = assistant_rollout(
            target_model,
            assistant_model,
            batch["input_ids"],
            batch["attention_mask"],
            return_target_logits=cfg.kl_weight > 0,
            steps=cfg.rollout_steps,
            pad_token_id=pad_token_id,
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
    assistant_model.train()
    mean_loss = torch.stack(losses).mean().item() if losses else 0.0
    return {
        "eval_loss": mean_loss,
        "eval_accuracy": (correct / total).item() if total.item() else 0.0,
        "eval_perplexity": math.exp(min(mean_loss, 20.0)),
    }


def save_checkpoint(
    accelerator: Accelerator,
    assistant_model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    cfg: Eagle3TrainingConfig,
    step: int,
    name: str,
    metrics: dict[str, float] | None = None,
) -> Path:
    path = Path(cfg.output_dir) / name
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(assistant_model)
        save_assistant_checkpoint(
            unwrapped,
            tokenizer,
            path,
            {"step": step, "config": cfg.to_dict(), "metrics": metrics or {}},
        )
    accelerator.wait_for_everyone()
    return path


def train(cfg: Eagle3TrainingConfig) -> None:
    set_seed(cfg.seed)
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        mixed_precision="bf16" if cfg.bf16 else "fp16" if cfg.fp16 else "no",
    )
    dtype = torch.bfloat16 if cfg.bf16 else torch.float16 if cfg.fp16 else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.target_model_path,
        trust_remote_code=cfg.trust_remote_code,
        use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    target_model = AutoModelForCausalLM.from_pretrained(
        cfg.target_model_path,
        dtype=dtype,
        trust_remote_code=cfg.trust_remote_code,
    )
    target_model.eval().requires_grad_(False)
    assistant_model = load_assistant_model(
        cfg.assistant_model_path,
        dtype=dtype,
        trust_remote_code=cfg.trust_remote_code,
        adapter_path=cfg.assistant_adapter_path,
        is_trainable=True,
    )
    if cfg.use_lora and not cfg.assistant_adapter_path:
        assistant_model = add_lora(
            assistant_model,
            rank=cfg.lora_rank,
            alpha=cfg.lora_alpha,
            dropout=cfg.lora_dropout,
        )
    if cfg.gradient_checkpointing:
        assistant_model.gradient_checkpointing_enable()
        assistant_model.config.use_cache = False

    train_loader = make_loader(cfg.train_data_dir, tokenizer, cfg, train=True)
    eval_loader = make_loader(cfg.eval_data_dir, tokenizer, cfg, train=False)
    if train_loader is None:
        raise FileNotFoundError(f"No train data found in {cfg.train_data_dir}")

    optimizer = AdamW(assistant_model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    steps_per_epoch = math.ceil(len(train_loader) / cfg.gradient_accumulation_steps)
    total_steps = cfg.max_steps if cfg.max_steps > 0 else steps_per_epoch * cfg.num_train_epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, cfg.warmup_steps, total_steps)
    target_model, assistant_model, optimizer, train_loader, scheduler = accelerator.prepare(
        target_model,
        assistant_model,
        optimizer,
        train_loader,
        scheduler,
    )
    if eval_loader is not None:
        eval_loader = accelerator.prepare(eval_loader)

    writer = None
    if accelerator.is_main_process:
        writer = SummaryWriter(cfg.tensorboard_log_dir or str(Path(cfg.output_dir) / "tensorboard"))
    progress = tqdm(total=total_steps, disable=not accelerator.is_main_process, desc="Fine-tuning assistant")
    best_eval_loss = float("inf")
    global_step = 0

    for _ in range(cfg.num_train_epochs):
        assistant_model.train()
        for batch in train_loader:
            if cfg.max_steps > 0 and global_step >= cfg.max_steps:
                break
            with accelerator.accumulate(assistant_model):
                assistant_logits, target_logits = assistant_rollout(
                    target_model,
                    assistant_model,
                    batch["input_ids"],
                    batch["attention_mask"],
                    return_target_logits=cfg.kl_weight > 0,
                    steps=cfg.rollout_steps,
                    pad_token_id=tokenizer.pad_token_id,
                )
                loss, metrics = compute_rollout_loss_and_metrics(
                    assistant_logits,
                    target_logits,
                    batch["labels"],
                    temperature=cfg.temperature,
                    kl_weight=cfg.kl_weight,
                    ce_weight=cfg.ce_weight,
                    decay=cfg.rollout_decay,
                )
                accelerator.backward(loss)
                if accelerator.sync_gradients and cfg.max_grad_norm > 0:
                    accelerator.clip_grad_norm_(assistant_model.parameters(), cfg.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                global_step += 1
                progress.update(1)
                if global_step % cfg.logging_steps == 0 and accelerator.is_main_process:
                    progress.set_postfix(loss=f"{metrics['loss']:.4f}", acc=f"{metrics['accuracy']:.4f}")
                    if writer:
                        for key, value in metrics.items():
                            writer.add_scalar(f"train/{key}", value, global_step)
                        writer.add_scalar("train/learning_rate", scheduler.get_last_lr()[0], global_step)
                if eval_loader is not None and cfg.eval_steps > 0 and global_step % cfg.eval_steps == 0:
                    eval_metrics = evaluate(
                        accelerator,
                        target_model,
                        assistant_model,
                        eval_loader,
                        cfg,
                        tokenizer.pad_token_id,
                    )
                    if eval_metrics["eval_loss"] < best_eval_loss:
                        best_eval_loss = eval_metrics["eval_loss"]
                        save_checkpoint(
                            accelerator,
                            assistant_model,
                            tokenizer,
                            cfg,
                            global_step,
                            "best-checkpoint",
                            eval_metrics,
                        )
                    if accelerator.is_main_process and writer:
                        for key, value in eval_metrics.items():
                            writer.add_scalar(key.replace("eval_", "eval/"), value, global_step)
                if cfg.save_steps > 0 and global_step % cfg.save_steps == 0:
                    save_checkpoint(
                        accelerator,
                        assistant_model,
                        tokenizer,
                        cfg,
                        global_step,
                        f"checkpoint-{global_step}",
                    )
        if cfg.max_steps > 0 and global_step >= cfg.max_steps:
            break

    progress.close()
    final_metrics = evaluate(
        accelerator,
        target_model,
        assistant_model,
        eval_loader,
        cfg,
        tokenizer.pad_token_id,
    )
    if not final_metrics or final_metrics["eval_loss"] < best_eval_loss:
        save_checkpoint(
            accelerator,
            assistant_model,
            tokenizer,
            cfg,
            global_step,
            "best-checkpoint",
            final_metrics,
        )
    save_checkpoint(accelerator, assistant_model, tokenizer, cfg, global_step, "final-assistant", final_metrics)
    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(assistant_model)
        if cfg.merge_lora_on_save and hasattr(unwrapped, "merge_and_unload"):
            merged = unwrapped.merge_and_unload()
            save_assistant_checkpoint(
                merged,
                tokenizer,
                Path(cfg.output_dir) / "final-assistant-merged",
                {"step": global_step, "config": cfg.to_dict(), "metrics": final_metrics},
            )
            print(f"Merged vLLM assistant saved under {Path(cfg.output_dir) / 'final-assistant-merged'}")
        if writer:
            writer.close()
        print(json.dumps(final_metrics, indent=2))
        print(f"Assistant saved under {Path(cfg.output_dir) / 'final-assistant'}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = Eagle3TrainingConfig.from_yaml(args.config)
    os.makedirs(cfg.output_dir, exist_ok=True)
    train(cfg)


if __name__ == "__main__":
    main()
