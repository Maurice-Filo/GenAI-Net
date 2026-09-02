#!/usr/bin/env bash

set -euo pipefail

ROOT="$(realpath -- "$(dirname -- "${BASH_SOURCE[0]}")/../..")"
CAMPAIGN_ROOT=/local0/home/rossin/ai-workspaces/deepseek-test/crn-runs/paper-campaigns
DSH_HOME=/local0/home/rossin/ai-workspaces/deepseek-test/.dsh-home
RAG_INDEX="$ROOT/literature_rag/index/literature.sqlite3"
PRO_STATUS="$CAMPAIGN_ROOT/20260821T-genai-net-llm-pro-logic-20seed-timeout3600-r4/status.json"
LAUNCHER="$ROOT/comparisons/llm_crn_generation/run_genai_net_llm_20seed_campaign.py"
RL_GPU_INDEX="${RL_GPU_INDEX:-1}"
RL_GPU_UUID="$(nvidia-smi --id="$RL_GPU_INDEX" --query-gpu=uuid --format=csv,noheader | tr -d ' ')"

log() {
    printf '[%s] %s\n' "$(date --iso-8601=seconds)" "$*"
}

pro_is_idle() {
    [[ -f "$PRO_STATUS" ]] || return 1
    grep -q '^  "active": \[\],$' "$PRO_STATUS" || return 1
    grep -q '^  "pending": \[\],$' "$PRO_STATUS"
}

run_campaign() {
    log "Launching campaign $1."
    "$ROOT/.venv/bin/python" "$LAUNCHER" \
        --campaign-id "$1" \
        --tasks logic rpa \
        --seeds 20 \
        --epochs "$2" \
        --rl-batch-size "$3" \
        --total-candidate-budget "$4" \
        --llm-candidates 10 \
        --llm-every 20 \
        --max-llm-in-flight 5 \
        --global-llm-concurrency 8 \
        --max-agent-evaluations 0 \
        --max-parallel 15 \
        --cpus-per-run 4 \
        --rl-gpu "$RL_GPU_UUID" \
        --llm-timeout 3600 \
        --model deepseek-v4-flash \
        --llm-provider deepseek-official \
        --method-name "$5" \
        --run-suffix "$6" \
        --dsh-home "$DSH_HOME" \
        --workspace-root "$CAMPAIGN_ROOT" \
        --comet-project genai-net-v4-flash-paper "${@:7}"
}

log "Queue active; waiting for the corrected Pro Logic campaign to become idle."
until pro_is_idle; do
    sleep 300
done

run_campaign \
    flash-long300-20seed 300 1023 307200 \
    genai_net_llm_flash_long300 cvode_llm_flash_long300

log "Launching deterministic breadth pilot (six tasks, five seeds each)."
"$ROOT/.venv/bin/python" "$LAUNCHER" \
    --campaign-id flash-breadth-deterministic-5seed \
    --tasks dose_hill dose_ultrasensitive dose_biphasic classifier oscillator_mean oscillator_frequency \
    --seeds 5 \
    --epochs 100 \
    --rl-batch-size 1023 \
    --total-candidate-budget 102400 \
    --llm-candidates 10 \
    --llm-every 20 \
    --max-llm-in-flight 5 \
    --global-llm-concurrency 8 \
    --max-agent-evaluations 0 \
    --max-parallel 15 \
    --cpus-per-run 4 \
    --rl-gpu "$RL_GPU_UUID" \
    --llm-timeout 3600 \
    --model deepseek-v4-flash \
    --llm-provider deepseek-official \
    --method-name genai_net_llm_flash_breadth \
    --run-suffix cvode_llm_flash_breadth \
    --dsh-home "$DSH_HOME" \
    --workspace-root "$CAMPAIGN_ROOT" \
    --comet-project genai-net-v4-flash-paper \
    --skip-postprocessing

ASSESSMENT="$CAMPAIGN_ROOT/flash-breadth-deterministic-5seed/consistency_assessment.json"
mapfile -t EXTENSION_TASKS < <(
    "$ROOT/.venv/bin/python" \
        "$ROOT/comparisons/llm_crn_generation/assess_breadth_consistency.py" \
        --raw-root "$ROOT/comparisons/rpa_search/data/raw" \
        --output "$ASSESSMENT" \
        --tasks-only
)
if ((${#EXTENSION_TASKS[@]})); then
    log "Extending inconsistent deterministic tasks to ten seeds: ${EXTENSION_TASKS[*]}."
    "$ROOT/.venv/bin/python" "$LAUNCHER" \
        --campaign-id flash-breadth-deterministic-extension-seeds5to9 \
        --tasks "${EXTENSION_TASKS[@]}" \
        --seeds 5 --seed-start 5 --epochs 100 \
        --rl-batch-size 1023 --total-candidate-budget 102400 \
        --llm-candidates 10 --llm-every 20 --max-llm-in-flight 5 \
        --global-llm-concurrency 8 \
        --max-agent-evaluations 0 --max-parallel 15 --cpus-per-run 4 --rl-gpu "$RL_GPU_UUID" \
        --llm-timeout 3600 --model deepseek-v4-flash \
        --llm-provider deepseek-official \
        --method-name genai_net_llm_flash_breadth \
        --run-suffix cvode_llm_flash_breadth \
        --dsh-home "$DSH_HOME" --workspace-root "$CAMPAIGN_ROOT" \
        --comet-project genai-net-v4-flash-paper --skip-postprocessing
else
    log "All deterministic breadth tasks met the five-seed consistency rule."
fi

log "Launching stochastic-RPA breadth pilot sequentially on the simulation GPU."
"$ROOT/.venv/bin/python" "$LAUNCHER" \
    --campaign-id flash-breadth-stochastic-rpa-5seed \
    --tasks stochastic_rpa \
    --seeds 5 \
    --epochs 100 \
    --rl-batch-size 159 \
    --total-candidate-budget 16000 \
    --llm-candidates 10 \
    --llm-every 20 \
    --max-llm-in-flight 1 \
    --global-llm-concurrency 8 \
    --max-agent-evaluations 0 \
    --max-parallel 1 \
    --cpus-per-run 4 \
    --rl-gpu "$RL_GPU_UUID" \
    --llm-timeout 3600 \
    --model deepseek-v4-flash \
    --llm-provider deepseek-official \
    --method-name genai_net_llm_flash_stochastic_rpa \
    --run-suffix ssa_llm_flash_breadth \
    --dsh-home "$DSH_HOME" \
    --workspace-root "$CAMPAIGN_ROOT" \
    --comet-project genai-net-v4-flash-paper \
    --skip-postprocessing

SSA_ASSESSMENT="$CAMPAIGN_ROOT/flash-breadth-stochastic-rpa-5seed/consistency_assessment.json"
mapfile -t SSA_EXTENSION < <(
    "$ROOT/.venv/bin/python" \
        "$ROOT/comparisons/llm_crn_generation/assess_breadth_consistency.py" \
        --raw-root "$ROOT/comparisons/rpa_search/data/raw" \
        --method genai_net_llm_flash_stochastic_rpa \
        --run-suffix ssa_llm_flash_breadth \
        --candidate-budget 16000 \
        --tasks stochastic_rpa \
        --output "$SSA_ASSESSMENT" \
        --tasks-only
)
if ((${#SSA_EXTENSION[@]})); then
    log "Extending stochastic RPA to ten seeds under the same consistency rule."
    "$ROOT/.venv/bin/python" "$LAUNCHER" \
        --campaign-id flash-breadth-stochastic-rpa-extension-seeds5to9 \
        --tasks stochastic_rpa --seeds 5 --seed-start 5 --epochs 100 \
        --rl-batch-size 159 --total-candidate-budget 16000 \
        --llm-candidates 10 --llm-every 20 --max-llm-in-flight 1 \
        --global-llm-concurrency 8 \
        --max-agent-evaluations 0 --max-parallel 1 --cpus-per-run 4 --rl-gpu "$RL_GPU_UUID" \
        --llm-timeout 3600 --model deepseek-v4-flash \
        --llm-provider deepseek-official \
        --method-name genai_net_llm_flash_stochastic_rpa \
        --run-suffix ssa_llm_flash_breadth \
        --dsh-home "$DSH_HOME" --workspace-root "$CAMPAIGN_ROOT" \
        --comet-project genai-net-v4-flash-paper --skip-postprocessing
else
    log "Stochastic RPA met the five-seed consistency rule."
fi

log "Primary and breadth cohorts completed; launching matched no-communication ablation."
run_campaign \
    flash-no-communication-long300-20seed 300 1023 307200 \
    genai_net_llm_flash_no_communication cvode_llm_flash_no_communication \
    --communication-mode none \
    --skip-postprocessing

log "Matched communication ablation completed; launching optional terminal studies."
run_campaign \
    flash-rag-20seed 100 1023 102400 \
    genai_net_llm_flash_rag cvode_llm_flash_rag \
    --literature-rag-index "$RAG_INDEX"

run_campaign \
    flash-exclusion-20seed 100 1021 102400 \
    genai_net_llm_flash_exclusion cvode_llm_flash_exclusion \
    --forbidden-topology-m 5 \
    --forbidden-topology-every 5 \
    --forbidden-optimization-max-evaluations 50 \
    --forbidden-optimization-timeout 120

log "All scheduled Flash cohorts completed."
