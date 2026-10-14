#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
: "${MODEL:?Set MODEL to the local Qwen3-4B-Thinking-2507 directory}"
args=(--model "$MODEL" --served-model-name base --host 127.0.0.1 --port "${PORT:-8001}"
      --max-model-len 32768 --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.85}"
      --tensor-parallel-size "${TP:-1}" --enable-reasoning --reasoning-parser deepseek_r1)
lora=()
for role in common generation review; do
  var="${role^^}_ADAPTER"
  if [[ -n "${!var:-}" ]]; then lora+=("$role=${!var}"); fi
done
if [[ ${#lora[@]} -gt 0 ]]; then
  args+=(--enable-lora --max-lora-rank 16 --max-loras "${#lora[@]}" --lora-modules "${lora[@]}")
fi
exec python -m vllm.entrypoints.openai.api_server "${args[@]}" "$@"
