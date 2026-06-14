from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .prompting import DEFAULT_SYSTEM_PROMPT, build_prompt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run vLLM speculative decoding with a Gemma 4 assistant checkpoint."
    )
    parser.add_argument("--target-model", required=True, help="Target model path or Hugging Face ID.")
    parser.add_argument(
        "--draft-model",
        required=True,
        help="Fine-tuned Gemma 4 *-assistant model path or Hugging Face ID.",
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--response-text", help="Generate one GenUI JSON response.")
    input_group.add_argument("--input-jsonl", help="Batch JSONL containing response_text fields.")
    parser.add_argument("--output-jsonl", help="Required with --input-jsonl.")
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--num-speculative-tokens", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--draft-tensor-parallel-size", type=int)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--max-model-len", type=int)
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=("auto", "float16", "half", "bfloat16", "float32", "float"),
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.input_jsonl and not args.output_jsonl:
        raise ValueError("--output-jsonl is required with --input-jsonl.")
    if args.output_jsonl and not args.input_jsonl:
        raise ValueError("--output-jsonl can only be used with --input-jsonl.")
    if args.num_speculative_tokens < 1:
        raise ValueError("--num-speculative-tokens must be at least 1.")
    if args.max_new_tokens < 1:
        raise ValueError("--max-new-tokens must be at least 1.")
    validate_draft_model(args.draft_model)


def validate_draft_model(draft_model: str) -> None:
    path = Path(draft_model)
    if not path.exists():
        return
    if not path.is_dir():
        raise ValueError("--draft-model must be a Hugging Face model directory or model ID.")

    config_path = path / "config.json"
    if not config_path.exists():
        raise ValueError(f"No config.json found in local draft model directory: {path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("model_type") not in {"gemma4_assistant", "gemma4_unified_assistant"}:
        raise ValueError(
            "Expected a Gemma 4 assistant checkpoint; config.json model_type is "
            f"{config.get('model_type')!r}."
        )


def build_speculative_config(args: argparse.Namespace) -> dict[str, Any]:
    config: dict[str, Any] = {
        "method": "mtp",
        "model": args.draft_model,
        "num_speculative_tokens": args.num_speculative_tokens,
    }
    if args.draft_tensor_parallel_size is not None:
        config["draft_tensor_parallel_size"] = args.draft_tensor_parallel_size
    return config


def build_llm_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "model": args.target_model,
        "dtype": args.dtype,
        "tensor_parallel_size": args.tensor_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "speculative_config": build_speculative_config(args),
        "trust_remote_code": args.trust_remote_code,
        "enforce_eager": args.enforce_eager,
        "seed": args.seed,
    }
    if args.max_model_len is not None:
        kwargs["max_model_len"] = args.max_model_len
    return kwargs


def load_batch(path: str, system_prompt: str) -> tuple[list[dict[str, Any]], list[str]]:
    records: list[dict[str, Any]] = []
    prompts: list[str] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if "response_text" not in record:
                raise ValueError(f"Missing response_text in {path} line {line_number}.")
            records.append(record)
            prompts.append(build_prompt(str(record["response_text"]), system_prompt))
    return records, prompts


def write_batch(
    output_path: str,
    records: list[dict[str, Any]],
    completions: list[str],
) -> None:
    if len(records) != len(completions):
        raise ValueError("Record and completion counts do not match.")
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for record, completion in zip(records, completions):
            output_record = dict(record)
            output_record["predicted_genui_json"] = completion.strip()
            handle.write(json.dumps(output_record, ensure_ascii=False) + "\n")


def run(args: argparse.Namespace) -> None:
    validate_args(args)

    from vllm import LLM, SamplingParams

    if args.response_text is not None:
        records = None
        prompts = [build_prompt(args.response_text, args.system_prompt)]
    else:
        records, prompts = load_batch(args.input_jsonl, args.system_prompt)

    llm = LLM(**build_llm_kwargs(args))
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_new_tokens,
    )
    outputs = llm.generate(prompts, sampling_params)
    completions = [output.outputs[0].text for output in outputs]

    if records is None:
        print(completions[0].strip())
    else:
        write_batch(args.output_jsonl, records, completions)
        print(f"Saved {len(completions)} predictions to {args.output_jsonl}")


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
