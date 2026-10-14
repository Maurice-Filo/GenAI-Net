#!/usr/bin/env bash
# Offline analysis and figure reproduction. Never launches an LLM/search job.
set -euo pipefail
release_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$release_root"
if [[ ! -x .venv/bin/python ]]; then
    python3 -m venv .venv
fi
.venv/bin/python -m pip install --disable-pip-version-check --no-index \
    --find-links wheels -r requirements-analysis.txt
exec .venv/bin/python reproduce.py "${@:-all}"
