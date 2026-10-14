#!/usr/bin/env bash
# Offline analysis and figure reproduction. Never launches an LLM/search job.
set -euo pipefail
script_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null && pwd)"
data_root="${FULLDUPLEX_DATA_ROOT:-$script_root}"
arguments=("$@")
for ((i=0; i<${#arguments[@]}; i++)); do
    if [[ ${arguments[i]} == --data-root ]]; then
        data_root="${arguments[i+1]:?Supply a directory after --data-root}"
    elif [[ ${arguments[i]} == --data-root=* ]]; then
        data_root="${arguments[i]#--data-root=}"
    fi
done
data_root="$(cd -- "$data_root" >/dev/null && pwd)"
if [[ ! -f "$data_root/manifest/runs.json" ]]; then
    printf 'Provide --data-root /path/to/extracted-release (or FULLDUPLEX_DATA_ROOT).\n' >&2
    exit 2
fi
cd "$script_root"
if [[ ! -x .venv/bin/python ]]; then
    python3 -m venv .venv
fi
.venv/bin/python -m pip install --disable-pip-version-check --no-index \
    --find-links "$data_root/wheels" -r requirements-analysis.txt
if (( $# == 0 )); then set -- all; fi
exec .venv/bin/python reproduce.py "$@" --data-root "$data_root"
