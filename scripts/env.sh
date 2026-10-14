#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export HOME="$ROOT/.cache/home" XDG_CACHE_HOME="$ROOT/.cache/xdg"
export HF_HOME="$ROOT/.cache/huggingface" PIP_CACHE_DIR="$ROOT/.cache/pip"
export CONDA_PKGS_DIRS="$ROOT/.cache/conda" UV_CACHE_DIR="$ROOT/.cache/uv"
export TRITON_CACHE_DIR="$ROOT/.cache/triton" VLLM_CACHE_ROOT="$ROOT/.cache/vllm"
export TORCHINDUCTOR_CACHE_DIR="$ROOT/.cache/torchinductor"
export TMPDIR="${TMPDIR_OVERRIDE:-/tmp/gsca_${UID}_${SLURM_JOB_ID:-$$}}"
export RAY_TMPDIR="$TMPDIR/ray" VLLM_RPC_BASE_PATH="$TMPDIR"
export SPREADSHEET_WORK_DIR="$ROOT/.runtime"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export PYTHONPATH="$ROOT/src" VLLM_USE_V1=0 NCCL_CUMEM_ENABLE=0
unset PYTORCH_CUDA_ALLOC_CONF PYTORCH_ALLOC_CONF TRANSFORMERS_CACHE
mkdir -p "$HOME" "$XDG_CACHE_HOME" "$TMPDIR" "$SPREADSHEET_WORK_DIR/scratch"
export ENV_PREFIX="${ENV_PREFIX:-$ROOT/.envs/agent}"
if [[ "${SKIP_ACTIVATE:-0}" != 1 ]]; then
  if [[ ! -x "$ENV_PREFIX/bin/python" ]]; then
    echo 'Create the environment first with scripts/setup_env.sh, or set ENV_PREFIX.' >&2
    exit 2
  fi
  export PATH="$ENV_PREFIX/bin:$PATH"
fi
cd "$ROOT"
