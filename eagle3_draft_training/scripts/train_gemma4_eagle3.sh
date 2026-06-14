#!/usr/bin/env bash
set -euo pipefail

echo "This launcher now fine-tunes the native Gemma 4 assistant checkpoint." >&2
exec bash "$(dirname "$0")/train_gemma4_assistant.sh" "$@"
