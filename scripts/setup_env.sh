#!/usr/bin/env bash
set -euo pipefail
CONDA_EXE_PATH="${CONDA_EXE:-$(command -v conda || true)}"
SKIP_ACTIVATE=1 source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
[[ -n "$CONDA_EXE_PATH" ]] || { echo 'Load conda first (or set CONDA_EXE).' >&2; exit 2; }
if [[ ! -x "$ENV_PREFIX/bin/python" ]]; then
  "$CONDA_EXE_PATH" create -y -p "$ENV_PREFIX" python=3.11
fi
export PATH="$ENV_PREFIX/bin:$PATH"
python -m pip install 'uv==0.8.22'
VERL_SRC="$ROOT/external/verl"
if [[ ! -d "$VERL_SRC/.git" ]]; then
  git clone --branch v0.6.0 --depth 1 https://github.com/verl-project/verl.git "$VERL_SRC"
fi
[[ "$(git -C "$VERL_SRC" describe --tags --exact-match HEAD)" == v0.6.0 ]]
# Resolve CUDA wheels as one constrained set; never silently upgrade to CUDA 13.
uv pip install --python "$ENV_PREFIX/bin/python" --torch-backend cu128 \
  -c "$ROOT/requirements-cu128.txt" -r "$ROOT/requirements-cu128.txt" \
  -r "$VERL_SRC/requirements.txt"
if git -C "$VERL_SRC" apply --check "$ROOT/patches/verl-v0.6.0.patch"; then
  git -C "$VERL_SRC" apply "$ROOT/patches/verl-v0.6.0.patch"
else
  git -C "$VERL_SRC" apply --reverse --check "$ROOT/patches/verl-v0.6.0.patch"
fi
python -m pip install --no-deps -e "$VERL_SRC"
python -m pip install -c "$ROOT/requirements-cu128.txt" -e "$ROOT"
python -m pip check
python -m spreadsheet_agent.check_env
