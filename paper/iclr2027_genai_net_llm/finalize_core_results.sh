#!/usr/bin/env bash

set -euo pipefail

PAPER_DIR="$(realpath -- "$(dirname -- "${BASH_SOURCE[0]}")")"
ROOT="$(realpath -- "$PAPER_DIR/../..")"
LOG=/local0/tmp/core-paper-finalize.log

exec >"$LOG" 2>&1

printf '[%s] Auditing all 160 matched primary endpoints.\n' "$(date --iso-8601=seconds)"
PYTHONPATH="$ROOT" MPLCONFIGDIR=/local0/tmp/mpl "$ROOT/.venv/bin/python" \
    "$PAPER_DIR/audit_paper_experiments.py"
MPLCONFIGDIR=/local0/tmp/mpl "$ROOT/.venv/bin/python" \
    "$PAPER_DIR/generate_primary_results.py"

printf '[%s] Reproducing communication and structural diagnostics.\n' "$(date --iso-8601=seconds)"
PYTHONPATH="$ROOT" MPLCONFIGDIR=/local0/tmp/mpl "$ROOT/.venv/bin/python" \
    "$PAPER_DIR/generate_communication_mechanism_analysis.py"

printf '[%s] Regenerating the initial-context evidence figure.\n' "$(date --iso-8601=seconds)"
MPLCONFIGDIR=/local0/tmp/mpl "$ROOT/.venv/bin/python" \
    "$ROOT/comparisons/rpa_search/scripts/plot_breadth_initial_hof_effect.py"
cp "$ROOT/comparisons/rpa_search/figures/breadth_initial_hof_withheld_effect.pdf" \
    "$PAPER_DIR/figures/breadth_initial_hof_effect.pdf"

printf '[%s] Regenerating manuscript-native figures.\n' "$(date --iso-8601=seconds)"
MPLCONFIGDIR=/local0/tmp/mpl "$ROOT/.venv/bin/python" \
    "$PAPER_DIR/generate_paper_figures.py"

printf '[%s] Compiling the manuscript.\n' "$(date --iso-8601=seconds)"
make -C "$PAPER_DIR"

printf '[%s] Strict primary analysis and manuscript build complete.\n' \
    "$(date --iso-8601=seconds)"
