#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
python -m unittest discover -s tests -v
for script in scripts/*.sh; do bash -n "$script"; done
