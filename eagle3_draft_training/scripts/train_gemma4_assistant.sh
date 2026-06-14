#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

# Edit this section.
ASSISTANT_MODEL_PATH="/home/c.kulkarni/hf_models/google/gemma-4-E2B-it-assistant"
TARGET_MODEL_PATH="/home/c.kulkarni/hf_models/google/gemma-4-E2B-it"
ASSISTANT_ADAPTER_PATH=""
TRAIN_JSONL="/path/to/train.jsonl"
TEST_JSONL=""

OUTPUT_DIR="outputs/gemma4_assistant_finetuned"
TOKENIZED_DATA_DIR="data/tokenized/gemma4"
CONFIG_PATH="configs/gemma4_runtime.yaml"

RUN_PREPARE_DATA="true"
RUN_TRAIN="true"
RUN_EVAL="true"
SAVE_TEST_PREDICTIONS="true"

MAX_LENGTH="2048"
EVAL_RATIO="0.05"
SEED="42"
LEARNING_RATE="0.00002"
GRADIENT_ACCUMULATION_STEPS="4"
PER_DEVICE_TRAIN_BATCH_SIZE="1"
PER_DEVICE_EVAL_BATCH_SIZE="1"
NUM_TRAIN_EPOCHS="1"
MAX_STEPS="-1"
LOGGING_STEPS="10"
EVAL_STEPS="200"
SAVE_STEPS="500"
BF16="true"
FP16="false"
USE_LORA="true"
MERGE_LORA_ON_SAVE="true"
LORA_RANK="16"
LORA_ALPHA="32"
LORA_DROPOUT="0.05"
KL_WEIGHT="0.0"
CE_WEIGHT="1.0"
TEMPERATURE="1.0"
ROLLOUT_STEPS="1"
ROLLOUT_DECAY="0.8"
TRUST_REMOTE_CODE="true"

bool_flag() {
  [[ "$1" == "true" || "$1" == "1" || "$1" == "yes" ]]
}

mkdir -p "$(dirname "$CONFIG_PATH")" "$OUTPUT_DIR" "$TOKENIZED_DATA_DIR"
cat > "$CONFIG_PATH" <<YAML
assistant_model_path: ${ASSISTANT_MODEL_PATH}
target_model_path: ${TARGET_MODEL_PATH}
assistant_adapter_path: ${ASSISTANT_ADAPTER_PATH:-null}
train_data_dir: ${TOKENIZED_DATA_DIR}/train
eval_data_dir: ${TOKENIZED_DATA_DIR}/eval
test_data_dir: ${TOKENIZED_DATA_DIR}/test
output_dir: ${OUTPUT_DIR}
tensorboard_log_dir: ${OUTPUT_DIR}/tensorboard
max_length: ${MAX_LENGTH}
learning_rate: ${LEARNING_RATE}
weight_decay: 0.0
warmup_steps: 100
gradient_accumulation_steps: ${GRADIENT_ACCUMULATION_STEPS}
per_device_train_batch_size: ${PER_DEVICE_TRAIN_BATCH_SIZE}
per_device_eval_batch_size: ${PER_DEVICE_EVAL_BATCH_SIZE}
num_train_epochs: ${NUM_TRAIN_EPOCHS}
max_steps: ${MAX_STEPS}
logging_steps: ${LOGGING_STEPS}
eval_steps: ${EVAL_STEPS}
save_steps: ${SAVE_STEPS}
seed: ${SEED}
temperature: ${TEMPERATURE}
kl_weight: ${KL_WEIGHT}
ce_weight: ${CE_WEIGHT}
rollout_steps: ${ROLLOUT_STEPS}
rollout_decay: ${ROLLOUT_DECAY}
bf16: ${BF16}
fp16: ${FP16}
gradient_checkpointing: true
trust_remote_code: ${TRUST_REMOTE_CODE}
use_lora: ${USE_LORA}
merge_lora_on_save: ${MERGE_LORA_ON_SAVE}
lora_rank: ${LORA_RANK}
lora_alpha: ${LORA_ALPHA}
lora_dropout: ${LORA_DROPOUT}
YAML

TRUST_ARG=()
bool_flag "$TRUST_REMOTE_CODE" && TRUST_ARG=(--trust_remote_code)
python scripts/configure_from_assistant.py \
  --assistant_model_path "$ASSISTANT_MODEL_PATH" \
  --target_model_path "$TARGET_MODEL_PATH" \
  --config "$CONFIG_PATH" \
  "${TRUST_ARG[@]}" \
  --overwrite

if bool_flag "$RUN_PREPARE_DATA"; then
  if [[ "$TRAIN_JSONL" == "/path/to/train.jsonl" || ! -f "$TRAIN_JSONL" ]]; then
    echo "TRAIN_JSONL is not set to a real file: $TRAIN_JSONL" >&2
    exit 1
  fi
  PREP_ARGS=(
    --target_model_path "$TARGET_MODEL_PATH"
    --train_jsonl "$TRAIN_JSONL"
    --output_dir "$TOKENIZED_DATA_DIR"
    --max_length "$MAX_LENGTH"
    --eval_ratio "$EVAL_RATIO"
    --seed "$SEED"
  )
  [[ -n "$TEST_JSONL" ]] && PREP_ARGS+=(--test_jsonl "$TEST_JSONL")
  bool_flag "$TRUST_REMOTE_CODE" && PREP_ARGS+=(--trust_remote_code)
  python scripts/prepare_jsonl.py "${PREP_ARGS[@]}"
fi

bool_flag "$RUN_TRAIN" && accelerate launch -m eagle3_draft.train --config "$CONFIG_PATH"

CHECKPOINT="${OUTPUT_DIR}/best-checkpoint"
if bool_flag "$RUN_EVAL"; then
  DATA_DIR="${TOKENIZED_DATA_DIR}/eval"
  [[ -n "$TEST_JSONL" ]] && DATA_DIR="${TOKENIZED_DATA_DIR}/test"
  python -m eagle3_draft.eval \
    --config "$CONFIG_PATH" \
    --checkpoint_path "$CHECKPOINT" \
    --data_dir "$DATA_DIR"
fi

if bool_flag "$SAVE_TEST_PREDICTIONS" && [[ -n "$TEST_JSONL" ]]; then
  python -m eagle3_draft.test_generation \
    --config "$CONFIG_PATH" \
    --checkpoint_path "$CHECKPOINT" \
    --input_jsonl "$TEST_JSONL" \
    --output_jsonl "${CHECKPOINT}/test_predictions.jsonl"
fi

echo "Assistant checkpoint: ${OUTPUT_DIR}/final-assistant"
echo "Merged vLLM checkpoint: ${OUTPUT_DIR}/final-assistant-merged"
