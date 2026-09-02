#!/usr/bin/env bash

set -euo pipefail

while tmux has-session -t paper-core-experiments 2>/dev/null; do
    sleep 60
done

sleep 5
exec "$(realpath -- "$(dirname -- "${BASH_SOURCE[0]}")/finalize_core_results.sh")"
