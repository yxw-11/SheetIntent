#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
# Both datasets share the normalized format. Use a NEW report directory.
exec python -m spreadsheet_recalc.evaluate "$@"
