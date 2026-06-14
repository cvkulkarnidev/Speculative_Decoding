# Gemma 4 assistant fine-tuning

This project fine-tunes an existing Gemma 4 `*-assistant` checkpoint. It no
longer creates a custom drafter architecture.

This is intentionally not a reimplementation of SafeAILab EAGLE-3. The
[official EAGLE repository](https://github.com/SafeAILab/EAGLE) trains a
separate feature-fusion network and recommends
[SpecForge](https://github.com/sgl-project/SpecForge) for production EAGLE-3
training. Gemma 4 `*-assistant` checkpoints instead use the native
Transformers/vLLM MTP interface with target hidden states and shared KV states.

The frozen target model supplies:

- its final hidden state;
- the token embeddings;
- the shared full-attention and sliding-attention KV states.

Those values are passed to `Gemma4AssistantForCausalLM` using the same interface
as Transformers assisted generation. The trained output therefore remains a
native Hugging Face Gemma 4 assistant checkpoint and can be used by Transformers
or vLLM MTP speculative decoding.

## Install

```bash
pip install -r requirements.txt
```

Use a Transformers version that includes `gemma4_assistant`.

`bitsandbytes` is not required for the default BF16/FP16 LoRA path. If an
incompatible `bitsandbytes` package is installed, the trainer bypasses PEFT's
optional bitsandbytes dispatch. Quantized 4-bit/8-bit training still requires a
working bitsandbytes/Triton combination.

## Configure and train

Edit the variables at the top of:

```text
scripts/train_gemma4_assistant.sh
```

At minimum set:

```bash
ASSISTANT_MODEL_PATH="/path/to/google/gemma-4-E2B-it-assistant"
TARGET_MODEL_PATH="/path/to/google/gemma-4-E2B-it"
TRAIN_JSONL="/path/to/train.jsonl"
TEST_JSONL="/path/to/test.jsonl"  # optional
```

Then run:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/train_gemma4_assistant.sh
```

The previous launcher name remains as a compatibility wrapper:

```bash
bash scripts/train_gemma4_eagle3.sh
```

## Data

JSONL rows use:

```json
{"response_text": "Create a chart", "genui_json": {"type": "chart"}}
```

Prompt tokens are masked, so the objective is computed only for
`genui_json` tokens.

## Training objective

The default distills the frozen target model into the assistant:

```yaml
kl_weight: 1.0
ce_weight: 0.0
```

This materializes full target-vocabulary logits and uses substantially more
VRAM. The assistant input is aligned with the native MTP contract: target
hidden state `h_t` is paired with token `x_(t+1)` to predict `x_(t+2)`.

Supervised cross-entropy can be mixed in by setting `ce_weight` above zero, but
the target model should already perform the task being accelerated. A draft
model trained away from the target distribution will have poor acceptance.

The official EAGLE-3 trainer performs seven recurrent training-time-test steps
and weights them by `0.8**step`. Gemma's native assistant is a different MTP
architecture, but this repository supports the analogous rollout:

```yaml
rollout_steps: 1
rollout_decay: 0.8
```

Start with one step. After confirming memory use and acceptance quality, try
`rollout_steps: 3`, then up to `7`. Each extra step adds another assistant
forward pass.

LoRA is enabled by default. Training outputs include:

```text
outputs/gemma4_assistant_finetuned/
  best-checkpoint/          # best full model or LoRA adapter
  final-assistant/          # final full model or LoRA adapter
  final-assistant-merged/   # merged model for vLLM when LoRA is enabled
```

To continue an existing LoRA adapter, set:

```bash
ASSISTANT_ADAPTER_PATH="/path/to/assistant-adapter"
```

## Evaluate

```bash
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

python -m eagle3_draft.eval \
  --config configs/gemma4_runtime.yaml \
  --checkpoint_path outputs/gemma4_assistant_finetuned/best-checkpoint \
  --data_dir data/tokenized/gemma4/test
```

## Transformers speculative generation

```bash
python -m eagle3_draft.test_generation \
  --config configs/gemma4_runtime.yaml \
  --checkpoint_path outputs/gemma4_assistant_finetuned/best-checkpoint \
  --response_text "Create a chart showing monthly revenue"
```

Batch prediction:

```bash
python -m eagle3_draft.test_generation \
  --config configs/gemma4_runtime.yaml \
  --checkpoint_path outputs/gemma4_assistant_finetuned/best-checkpoint \
  --input_jsonl /path/to/test.jsonl \
  --output_jsonl outputs/test_predictions.jsonl
```

## vLLM MTP speculative decoding

Install vLLM:

```bash
pip install -r requirements-vllm.txt
```

Edit and run:

```bash
bash scripts/run_vllm_assistant.sh
```

Direct invocation:

```bash
python scripts/vllm_assistant_infer.py \
  --target-model /path/to/gemma-4-E2B-it \
  --draft-model outputs/gemma4_assistant_finetuned/final-assistant-merged \
  --input-jsonl /path/to/test.jsonl \
  --output-jsonl outputs/vllm_predictions.jsonl \
  --num-speculative-tokens 3 \
  --dtype bfloat16 \
  --trust-remote-code
```

The vLLM configuration uses:

```python
{"method": "mtp", "model": "/path/to/fine-tuned-assistant"}
```

Do not pass an unmerged LoRA adapter to vLLM. Use `final-assistant-merged`.
