#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml
from transformers import AutoConfig

from eagle3_draft.compatibility import SUPPORTED_ASSISTANT_MODEL_TYPES
from eagle3_draft.config_utils import get_hidden_size, get_vocab_size


def inspect_models(
    target_model_path: str,
    assistant_model_path: str,
    trust_remote_code: bool,
) -> dict[str, Any]:
    target = AutoConfig.from_pretrained(target_model_path, trust_remote_code=trust_remote_code)
    assistant = AutoConfig.from_pretrained(assistant_model_path, trust_remote_code=trust_remote_code)
    if assistant.model_type not in SUPPORTED_ASSISTANT_MODEL_TYPES:
        raise ValueError(
            f"{assistant_model_path} has model_type={assistant.model_type!r}; expected "
            f"one of {sorted(SUPPORTED_ASSISTANT_MODEL_TYPES)}."
        )
    target_vocab = get_vocab_size(target)
    assistant_vocab = get_vocab_size(assistant)
    target_hidden = get_hidden_size(target)
    backbone_hidden = int(getattr(assistant, "backbone_hidden_size", target_hidden))
    if target_vocab != assistant_vocab:
        raise ValueError(f"Target vocab size {target_vocab} != assistant vocab size {assistant_vocab}.")
    if target_hidden != backbone_hidden:
        raise ValueError(
            f"Target hidden size {target_hidden} != assistant backbone_hidden_size {backbone_hidden}."
        )
    return {
        "target_model_type": target.model_type,
        "assistant_model_type": assistant.model_type,
        "target_hidden_size": target_hidden,
        "assistant_backbone_hidden_size": backbone_hidden,
        "vocab_size": target_vocab,
        "recommended": {
            "assistant_model_path": assistant_model_path,
            "target_model_path": target_model_path,
        },
    }


def update_yaml_config(config_path: Path, model_info: dict[str, Any]) -> None:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    config.update(model_info["recommended"])
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_model_path", required=True)
    parser.add_argument("--assistant_model_path", required=True)
    parser.add_argument("--config", default="configs/gemma4_example.yaml")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    info = inspect_models(args.target_model_path, args.assistant_model_path, args.trust_remote_code)
    print(yaml.safe_dump(info, sort_keys=False))
    if args.overwrite:
        update_yaml_config(Path(args.config), info)
        print(f"Updated {args.config} with compatible target and assistant paths.")


if __name__ == "__main__":
    main()
