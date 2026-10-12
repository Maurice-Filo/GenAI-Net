#!/usr/bin/env python3
"""Wait for the local Qwen3.8 endpoint, then run one matched RPA cohort member."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CAMPAIGN_ID = "qwen38-local-v2-rpa-seed9000-100epoch-20260902-r8"
ENDPOINT = "http://127.0.0.1:18082/v1"
RL_GPU = "GPU-e28480e9-6ce2-60d6-887a-7c2a0cbd1e19"
DSH_HOME = Path.home() / "ai-workspaces/deepseek-test/.dsh-home"
CAMPAIGN_ROOT = Path.home() / "ai-workspaces/deepseek-test/crn-runs/paper-campaigns"


def read_models(endpoint: str, timeout: float = 5.0) -> list[str]:
    with urllib.request.urlopen(f"{endpoint}/models", timeout=timeout) as response:
        payload = json.load(response)
    records = payload.get("data") or payload.get("models") or []
    identifiers = []
    for record in records:
        identifier = record.get("id") or record.get("model") or record.get("name")
        if identifier:
            identifiers.append(str(identifier))
    return identifiers


def require_qwen38_model(endpoint: str) -> str:
    identifiers = read_models(endpoint)
    if len(identifiers) != 1:
        raise RuntimeError(
            f"Expected exactly one model at {endpoint}; received {identifiers!r}."
        )
    model = identifiers[0]
    normalized = model.lower().replace("-", "").replace("_", "")
    if "qwen" not in normalized or "3.8" not in normalized:
        raise RuntimeError(f"Endpoint model is not the expected Qwen3.8 model: {model!r}.")
    return model


def tool_probe_payload(model: str) -> dict[str, object]:
    return {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": "Call the noop function now. Do not answer in plain text.",
            }
        ],
        "temperature": 0,
        "max_tokens": 1024,
        "chat_template_kwargs": {"enable_thinking": True},
        "thinking_token_budget": 256,
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "noop",
                    "description": "A harmless compatibility test that takes no arguments.",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "additionalProperties": False,
                    },
                },
            }
        ],
        "tool_choice": "auto",
    }


def probe_tool_call(endpoint: str, model: str, timeout: float = 120.0) -> None:
    request = urllib.request.Request(
        f"{endpoint}/chat/completions",
        data=json.dumps(tool_probe_payload(model)).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.load(response)
    choices = payload.get("choices") or []
    if not choices:
        raise RuntimeError("The Qwen endpoint returned no tool-probe choices.")
    choice = choices[0]
    tool_calls = (choice.get("message") or {}).get("tool_calls") or []
    if not tool_calls:
        raise RuntimeError("The Qwen endpoint did not emit an automatic tool call.")
    function = tool_calls[0].get("function") or {}
    if function.get("name") != "noop":
        raise RuntimeError(
            f"The Qwen endpoint called an unexpected tool: {function.get('name')!r}."
        )
    arguments = function.get("arguments", "")
    try:
        parsed_arguments = json.loads(arguments) if isinstance(arguments, str) else arguments
    except json.JSONDecodeError as error:
        raise RuntimeError("The Qwen endpoint emitted invalid JSON tool arguments.") from error
    if parsed_arguments != {}:
        raise RuntimeError(
            f"The Qwen endpoint emitted unexpected noop arguments: {parsed_arguments!r}."
        )
    if choice.get("finish_reason") != "tool_calls":
        raise RuntimeError(
            "The Qwen endpoint did not finish the probe with finish_reason='tool_calls'."
        )


def campaign_command(model: str) -> list[str]:
    return [
        str(ROOT / ".venv/bin/python"),
        str(ROOT / "comparisons/llm_crn_generation/run_genai_net_llm_20seed_campaign.py"),
        "--campaign-id",
        CAMPAIGN_ID,
        "--tasks",
        "rpa",
        "--max-parallel",
        "1",
        "--cpus-per-run",
        "4",
        "--rl-gpu",
        RL_GPU,
        "--seeds",
        "1",
        "--seed-start",
        "9000",
        "--epochs",
        "100",
        "--rl-batch-size",
        "1023",
        "--total-candidate-budget",
        "102400",
        "--llm-candidates",
        "10",
        "--llm-every",
        "20",
        "--max-llm-in-flight",
        "5",
        "--max-agent-evaluations",
        "0",
        "--llm-timeout",
        "3600",
        "--global-llm-concurrency",
        "8",
        "--comet-project",
        "genai-net-local-llm-paper",
        "--campaign-stage",
        "sentinel",
        "--model",
        model,
        "--task-prompt-variant",
        "standard",
        "--generation-backend",
        "harness",
        "--llm-provider",
        "local-qwen38-18081",
        "--llm-base-url",
        ENDPOINT,
        "--local-llm-context-window",
        "262144",
        "--local-llm-max-output-tokens",
        "8192",
        "--communication-mode",
        "full",
        "--withhold-initial-hof",
        "--candidate-validation-policy",
        "independent-members",
        "--method-name",
        "genai_net_llm_qwen38_local_contract_v2_rpa_r8",
        "--run-suffix",
        "cvode_llm_qwen38_local_contract_v2_rpa_r8",
        "--dsh-home",
        str(DSH_HOME),
        "--workspace-root",
        str(CAMPAIGN_ROOT),
        "--skip-postprocessing",
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wait-seconds", type=float, default=21600.0)
    parser.add_argument("--poll-seconds", type=float, default=10.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    deadline = time.monotonic() + args.wait_seconds
    while True:
        try:
            model = require_qwen38_model(ENDPOINT)
            probe_tool_call(ENDPOINT, model)
            break
        except (OSError, RuntimeError, urllib.error.URLError) as error:
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Qwen3.8 endpoint did not pass preflight within {args.wait_seconds}s."
                ) from error
            print(f"[wait] {ENDPOINT}: {error}", flush=True)
            time.sleep(args.poll_seconds)

    print(f"[ready] model={model}; launching {CAMPAIGN_ID}", flush=True)
    environment = os.environ.copy()
    environment.pop("COMET_API_KEY", None)
    environment.update(
        {
            "PYTHONPATH": str(ROOT),
            "MPLCONFIGDIR": "/local0/tmp/mpl",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        }
    )
    subprocess.run(campaign_command(model), cwd=ROOT, env=environment, check=True)


if __name__ == "__main__":
    main()
