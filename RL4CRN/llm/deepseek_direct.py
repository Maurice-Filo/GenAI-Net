"""Auditable one-shot DeepSeek generation without thinking or agent tools."""

from __future__ import annotations

import fcntl
import json
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Mapping, Optional
from uuid import uuid4

import yaml

from RL4CRN.llm.generator import LLMCRNGenerator, LLMGenerationRound
from RL4CRN.llm.prompts import format_reaction_library
from RL4CRN.llm.schemas import LLMGenerationConfig, parse_candidates_payload


DIRECT_SYSTEM_PROMPT = (
    "You propose chemical reaction networks for an externally evaluated optimization task. "
    "Return only the requested JSON object. Do not explain your answer, call tools, simulate, "
    "or include reasoning fields."
)


class DirectDeepSeekClient:
    """Single-call Chat Completions client with provider thinking disabled."""

    def __init__(
        self,
        *,
        workspace_root: str | Path,
        dsh_home: str | Path,
        model: str = "deepseek-v4-flash",
        base_url: str = "https://api.deepseek.com",
        timeout_seconds: float = 900.0,
        global_concurrency: int = 8,
    ):
        self.workspace_root = Path(workspace_root).expanduser().resolve()
        self.dsh_home = Path(dsh_home).expanduser().resolve()
        self.model = str(model)
        self.base_url = str(base_url).rstrip("/")
        self.timeout_seconds = float(timeout_seconds)
        self.global_concurrency = int(global_concurrency)
        if self.global_concurrency <= 0:
            raise ValueError("Direct DeepSeek global concurrency must be positive.")
        if self.base_url != "https://api.deepseek.com":
            raise ValueError("Direct DeepSeek requests must use the official HTTPS API endpoint.")

    def fork(self) -> "DirectDeepSeekClient":
        return type(self)(
            workspace_root=self.workspace_root,
            dsh_home=self.dsh_home,
            model=self.model,
            base_url=self.base_url,
            timeout_seconds=self.timeout_seconds,
            global_concurrency=self.global_concurrency,
        )

    def generate_json(
        self,
        prompt: str,
        *,
        generation_config: Optional[LLMGenerationConfig] = None,
    ) -> Dict[str, Any]:
        config = generation_config or LLMGenerationConfig()
        request_root = self._request_root()
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": DIRECT_SYSTEM_PROMPT},
                {"role": "user", "content": str(prompt)},
            ],
            "response_format": {"type": "json_object"},
            "thinking": {"type": "disabled"},
            "temperature": float(config.temperature),
        }
        if config.max_output_tokens is not None:
            payload["max_tokens"] = int(config.max_output_tokens)
        (request_root / "request.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self._api_key()}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        started = time.perf_counter()
        try:
            with self._global_request_slot() as slot:
                with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                    raw_response = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:2000]
            self._write_status(request_root, "http_error", started, error=detail)
            raise RuntimeError(f"DeepSeek direct request failed with HTTP {exc.code}: {detail}") from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            self._write_status(request_root, "transport_error", started, error=str(exc))
            raise RuntimeError(f"DeepSeek direct request failed: {exc}") from exc

        message = raw_response["choices"][0]["message"]
        reasoning = message.get("reasoning_content")
        if reasoning:
            self._write_status(
                request_root,
                "unexpected_reasoning",
                started,
                error="Provider returned reasoning_content despite thinking being disabled.",
            )
            raise RuntimeError("DeepSeek returned reasoning content in non-thinking mode.")
        content = message.get("content")
        if not isinstance(content, str):
            raise RuntimeError("DeepSeek direct response did not contain textual JSON content.")
        parsed = json.loads(content)
        if not isinstance(parsed, dict):
            raise ValueError("DeepSeek direct JSON response must be an object.")

        audit_response = {
            "id": raw_response.get("id"),
            "model": raw_response.get("model"),
            "usage": raw_response.get("usage"),
            "finish_reason": raw_response["choices"][0].get("finish_reason"),
            "reasoning_content_present": bool(reasoning),
            "content": parsed,
        }
        (request_root / "response.json").write_text(
            json.dumps(audit_response, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        self._write_status(request_root, "completed", started, slot=slot)
        return parsed

    def _request_root(self) -> Path:
        self.workspace_root.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        path = self.workspace_root / f"{stamp}-{uuid4().hex[:8]}"
        path.mkdir()
        return path

    def _api_key(self) -> str:
        credentials = self.dsh_home / ".credentials.yaml"
        data = yaml.safe_load(credentials.read_text(encoding="utf-8")) or {}
        key = str(data.get("DEEPSEEK_API_KEY", "")).strip()
        if not key:
            raise RuntimeError("DEEPSEEK_API_KEY is missing from the protected DSH credentials.")
        return key

    @contextmanager
    def _global_request_slot(self) -> Iterator[int]:
        slot_root = self.dsh_home / "direct-request-slots"
        slot_root.mkdir(parents=True, exist_ok=True)
        while True:
            for slot in range(self.global_concurrency):
                handle = (slot_root / f"slot-{slot:03d}.lock").open("a+")
                try:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    handle.close()
                    continue
                try:
                    yield slot
                finally:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                    handle.close()
                return
            time.sleep(0.1)

    @staticmethod
    def _write_status(
        path: Path,
        status: str,
        started: float,
        *,
        slot: Optional[int] = None,
        error: Optional[str] = None,
    ) -> None:
        payload = {
            "status": status,
            "duration_seconds": time.perf_counter() - started,
            "thinking": "disabled",
            "agent_tools": False,
            "slot": slot,
            "error": error,
        }
        (path / "status.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )


class DirectDeepSeekCRNGenerator(LLMCRNGenerator):
    """Simple prompt, one provider call, then canonical external evaluation."""

    client: DirectDeepSeekClient

    def fork(self) -> "DirectDeepSeekCRNGenerator":
        return type(self)(
            client=self.client.fork(),
            evaluator=self.evaluator,
            memory=deepcopy(self.memory),
            system_prompt=self.system_prompt,
            generation_config=self.generation_config,
        )

    def run_round(
        self,
        *,
        task_description: str,
        num_candidates: int = 10,
        hall_of_fame_iter: Optional[Iterable[Any]] = None,
        add_to_hall_of_fame: Optional[Any] = None,
        jsonl_path: Optional[str | Path] = None,
        logger: Any = None,
        step: Optional[int] = None,
        **_: Any,
    ) -> LLMGenerationRound:
        prompt = _direct_prompt(
            task_description=task_description,
            reaction_library=self.evaluator.library,
            max_added_reactions=self.evaluator.max_added_reactions,
            num_candidates=num_candidates,
            hall_of_fame_iter=hall_of_fame_iter,
        )
        raw_payload = self.client.generate_json(prompt, generation_config=self.generation_config)
        candidates = parse_candidates_payload(raw_payload)
        if len(candidates) != int(num_candidates):
            raise ValueError(
                f"Direct prompt requested {num_candidates} candidates but received {len(candidates)}."
            )
        evaluations = self.evaluator.evaluate_many(
            candidates,
            add_to_hall_of_fame=add_to_hall_of_fame,
            jsonl_path=jsonl_path,
        )
        self.memory.update_many(evaluations)
        if logger is not None and hasattr(logger, "log_metric"):
            logger.log_metric("LLM/Model Requests", 1, step=step)
            logger.log_metric("LLM/Workspace Tool Evaluations", 0, step=step)
            logger.log_metric("LLM/Thinking Enabled", 0, step=step)
        return LLMGenerationRound(
            prompt=prompt,
            candidates=candidates,
            evaluations=evaluations,
            raw_payload=raw_payload,
        )


def _direct_prompt(
    *,
    task_description: str,
    reaction_library: Any,
    max_added_reactions: int,
    num_candidates: int,
    hall_of_fame_iter: Optional[Iterable[Any]],
) -> str:
    hall = list(hall_of_fame_iter or ())[:5]
    if hall:
        entries = []
        for rank, env in enumerate(hall, start=1):
            state = env.state
            info = dict(getattr(state, "last_task_info", {}) or {})
            actions = []
            for action in getattr(state, "raw_actions_taken", ()) or ():
                actions.append(
                    {
                        "reaction_id": action.get("reaction index"),
                        "parameter_values": action.get(
                            "parameters", action.get("continuous parameters")
                        ),
                    }
                )
            entries.append({"rank": rank, "loss": info.get("reward"), "actions": actions})
            entries[-1]["crn"] = str(state)
        hof_text = json.dumps(entries, sort_keys=True)
    else:
        hof_text = "No Hall-of-Fame entries are provided for this request."
    return f"""Task:
{task_description}

Choose reactions only from this library:
{format_reaction_library(reaction_library)}

Current Hall of Fame, ranked by lower loss:
{hof_text}

Generate exactly {int(num_candidates)} distinct CRNs. Each CRN must use exactly
{int(max_added_reactions)} distinct reaction IDs. Use positive finite parameter values and
provide the complete parameter vector required by every selected reaction. Balance new
reaction-ID sets with parameter refinements when a Hall of Fame is present.

Return only this JSON shape, with no additional keys or prose:
{{"candidates":[{{"reaction_ids":[0,1],"parameter_values":[[1.0],[1.0]]}}]}}"""
