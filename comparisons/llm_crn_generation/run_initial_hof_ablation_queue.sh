#!/usr/bin/env bash

set -euo pipefail

ROOT="$(realpath -- "$(dirname -- "${BASH_SOURCE[0]}")/../..")"
CAMPAIGN_ROOT=/local0/home/rossin/ai-workspaces/deepseek-test/crn-runs/paper-campaigns
DSH_HOME=/local0/home/rossin/ai-workspaces/deepseek-test/.dsh-home
LAUNCHER="$ROOT/comparisons/llm_crn_generation/run_genai_net_llm_20seed_campaign.py"
ANALYZER="$ROOT/comparisons/rpa_search/scripts/analyze_initial_hof_ablation.py"
WAIT_PID="${WAIT_PID:-}"
WAIT_PIDS="${WAIT_PIDS:-}"
RL_GPU_INDEX="${RL_GPU_INDEX:-1}"
INITIAL_ID=flash-initial-hof-shown-20epoch-5seed
WITHHELD_ID=flash-initial-hof-withheld-20epoch-5seed

if [[ -n "$WAIT_PID" ]]; then
    printf '[%s] Waiting for predecessor queue PID %s.\n' "$(date --iso-8601=seconds)" "$WAIT_PID"
    while kill -0 "$WAIT_PID" 2>/dev/null; do
        sleep 300
    done
fi

if [[ -n "$WAIT_PIDS" ]]; then
    printf '[%s] Waiting for active worker PIDs: %s.\n' \
        "$(date --iso-8601=seconds)" "$WAIT_PIDS"
    IFS=',' read -r -a worker_pids <<< "$WAIT_PIDS"
    for worker_pid in "${worker_pids[@]}"; do
        while kill -0 "$worker_pid" 2>/dev/null; do
            sleep 60
        done
    done
fi

SOURCE_SNAPSHOT="$CAMPAIGN_ROOT/initial-hof-ablation-source.sha256"
find "$ROOT/RL4CRN" "$ROOT/comparisons/llm_crn_generation" "$ROOT/comparisons/rpa_search/src" \
    -type f -name '*.py' -print0 \
    | sort -z \
    | xargs -0 sha256sum > "$SOURCE_SNAPSHOT"

RL_GPU_UUID="$(nvidia-smi --id="$RL_GPU_INDEX" --query-gpu=uuid --format=csv,noheader | tr -d ' ')"
COMMON=(
    --tasks rpa logic
    --seeds 5
    --epochs 20
    --rl-batch-size 1023
    --total-candidate-budget 20480
    --llm-candidates 10
    --llm-every 20
    --max-llm-in-flight 1
    --global-llm-concurrency 8
    --max-agent-evaluations 0
    --max-parallel 5
    --cpus-per-run 4
    --rl-gpu "$RL_GPU_UUID"
    --llm-timeout 3600
    --model deepseek-v4-flash
    --llm-provider deepseek-official
    --communication-mode full
    --dsh-home "$DSH_HOME"
    --workspace-root "$CAMPAIGN_ROOT"
    --comet-project genai-net-v4-flash-paper
    --skip-postprocessing
)

printf '[%s] Launching paired initial-HOF context arms concurrently.\n' "$(date --iso-8601=seconds)"
"$ROOT/.venv/bin/python" "$LAUNCHER" \
    --campaign-id "$INITIAL_ID" \
    --method-name genai_net_llm_flash_initial_hof \
    --run-suffix cvode_flash_initial_hof \
    "${COMMON[@]}" &
INITIAL_PID=$!

"$ROOT/.venv/bin/python" "$LAUNCHER" \
    --campaign-id "$WITHHELD_ID" \
    --method-name genai_net_llm_flash_initial_context_free \
    --run-suffix cvode_flash_initial_context_free \
    --withhold-initial-hof \
    "${COMMON[@]}" &
WITHHELD_PID=$!

INITIAL_STATUS=0
WITHHELD_STATUS=0
wait "$INITIAL_PID" || INITIAL_STATUS=$?
wait "$WITHHELD_PID" || WITHHELD_STATUS=$?
if ((INITIAL_STATUS != 0 || WITHHELD_STATUS != 0)); then
    printf 'Initial-HOF arms failed: shown=%s withheld=%s\n' "$INITIAL_STATUS" "$WITHHELD_STATUS" >&2
    exit 1
fi

MPLCONFIGDIR=/local0/tmp/mpl "$ROOT/.venv/bin/python" "$ANALYZER" \
    --initial-hof-root "$CAMPAIGN_ROOT/$INITIAL_ID" \
    --context-free-root "$CAMPAIGN_ROOT/$WITHHELD_ID" \
    --initial-hof-status "$CAMPAIGN_ROOT/$INITIAL_ID/status.json" \
    --context-free-status "$CAMPAIGN_ROOT/$WITHHELD_ID/status.json" \
    --output "$ROOT/comparisons/rpa_search/figures/initial_hof_ablation_20epoch_5seed.pdf"

printf '[%s] Initial-HOF context ablation and analysis completed.\n' "$(date --iso-8601=seconds)"
