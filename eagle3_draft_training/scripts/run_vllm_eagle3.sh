#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

TARGET_MODEL="/home/c.kulkarni/hf_models/google/gemma-4-E2B-it"
DRAFT_MODEL="/path/to/vllm-compatible-eagle3-draft"

INPUT_JSONL="/path/to/test.jsonl"
OUTPUT_JSONL="outputs/vllm_eagle3_predictions.jsonl"

NUM_SPECULATIVE_TOKENS="3"
MAX_NEW_TOKENS="256"
TENSOR_PARALLEL_SIZE="1"
GPU_MEMORY_UTILIZATION="0.90"
DTYPE="bfloat16"

python scripts/vllm_eagle3_infer.py \
  --target-model "$TARGET_MODEL" \
  --draft-model "$DRAFT_MODEL" \
  --input-jsonl "$INPUT_JSONL" \
  --output-jsonl "$OUTPUT_JSONL" \
  --num-speculative-tokens "$NUM_SPECULATIVE_TOKENS" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --dtype "$DTYPE" \
  --trust-remote-code
