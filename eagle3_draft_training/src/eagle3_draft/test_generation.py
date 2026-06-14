from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import Eagle3TrainingConfig
from .eval import load_checkpoint
from .prompting import DEFAULT_SYSTEM_PROMPT, build_prompt


@torch.no_grad()
def generate(
    target_model: torch.nn.Module,
    assistant_model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    response_text: str,
    max_new_tokens: int,
    system_prompt: str,
) -> str:
    device = next(target_model.parameters()).device
    inputs = tokenizer(build_prompt(response_text, system_prompt), return_tensors="pt").to(device)
    generated = target_model.generate(
        **inputs,
        assistant_model=assistant_model,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )
    new_tokens = generated[0, inputs["input_ids"].shape[1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--response_text")
    parser.add_argument("--input_jsonl")
    parser.add_argument("--output_jsonl")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--system_prompt", default=DEFAULT_SYSTEM_PROMPT)
    args = parser.parse_args()

    cfg = Eagle3TrainingConfig.from_yaml(args.config)
    dtype = torch.bfloat16 if cfg.bf16 else torch.float16 if cfg.fp16 else torch.float32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(cfg.target_model_path, trust_remote_code=cfg.trust_remote_code)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    target_model = AutoModelForCausalLM.from_pretrained(
        cfg.target_model_path,
        dtype=dtype,
        trust_remote_code=cfg.trust_remote_code,
    ).eval().to(device)
    assistant_model = load_checkpoint(cfg, args.checkpoint_path, dtype).eval().to(device)

    if args.response_text:
        print(
            generate(
                target_model,
                assistant_model,
                tokenizer,
                args.response_text,
                args.max_new_tokens,
                args.system_prompt,
            )
        )
        return
    if not args.input_jsonl or not args.output_jsonl:
        raise ValueError("Pass --response_text or both --input_jsonl and --output_jsonl.")
    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(args.input_jsonl, "r", encoding="utf-8") as source, open(
        output_path, "w", encoding="utf-8"
    ) as destination:
        for line in source:
            record = json.loads(line)
            record["predicted_genui_json"] = generate(
                target_model,
                assistant_model,
                tokenizer,
                str(record["response_text"]),
                args.max_new_tokens,
                args.system_prompt,
            )
            destination.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
