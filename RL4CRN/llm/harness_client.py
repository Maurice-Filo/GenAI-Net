"""DeepSeek Harness client with isolated, auditable per-run workspaces."""

from __future__ import annotations

import hashlib
import fcntl
import json
import math
import os
import shlex
import shutil
import subprocess
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, Mapping, Optional, Sequence
from urllib.parse import urlparse
from uuid import uuid4

from RL4CRN.llm.benchmark_prompts import CRN_AGENT_SYSTEM_PROMPT
from RL4CRN.llm.schemas import LLMGenerationConfig


DEFAULT_DSH_PACKAGE = "@deepseek-ai/dsh@0.1.0-rc.8"

HARNESS_COMMON_TASK_TEMPLATE = (
    "Work only inside the current run workspace. Do not inspect or modify parent "
    "directories. Do not invoke Bash; use native workspace file tools. Read "
    "SYSTEM_PROMPT.md, TASK.md, the files under CONTEXT/, and then "
    "{relative_request}. Use targeted grep queries on REACTION_LIBRARY.tsv for "
    "reaction patterns. Read the project skill under .dsh/skills/ only when simulation "
    "evidence could improve the decision. Use no more than eight workspace-tool calls, "
    "then answer. Update REASONING_NOTES.md with a short, readable scientific decision "
    "summary. "
)
HARNESS_WRITER_TASK_SUFFIX = (
    "This is the Writer call. Read OUTPUT_GUIDE.json and DECIDER_DESIGNS.md; do not "
    "read OUTPUT_CONTRACT.json directly. Preserve the Decider's concrete scientific "
    "choices while implementing constraints and encoding them. Write the exact "
    "complete JSON answer to FINAL_RESPONSE.json as your final workspace action. "
    "Return only that JSON document, with no Markdown fences or commentary."
)
HARNESS_DECIDER_TASK_SUFFIX = (
    "This is the Decider call. Choose the requested concrete CRN structures and "
    "intended rates yourself. You may use any concise scientific notation. Do not "
    "read OUTPUT_GUIDE.json, do not emit machine JSON, and do not defer scientific "
    "choices to the Writer. Write the design record to DECIDER_DESIGNS.md as your "
    "final workspace action, then return the same concise text."
)


def build_harness_bot_task(response_kind: str, relative_request: str) -> str:
    """Return the role-specific Harness wrapper presented to one model call."""

    common = HARNESS_COMMON_TASK_TEMPLATE.format(relative_request=relative_request)
    if response_kind == "json":
        return common + HARNESS_WRITER_TASK_SUFFIX
    if response_kind == "text":
        return common + HARNESS_DECIDER_TASK_SUFFIX
    raise ValueError(f"Unsupported Harness response kind: {response_kind!r}")


class HarnessUnavailableError(RuntimeError):
    """Raised when the configured Harness command cannot be started."""


class HarnessResponseError(ValueError):
    """Raised when Harness does not return the requested response shape."""


@dataclass
class HarnessRunWorkspace:
    """Files and metadata belonging to one auditable Harness run."""

    path: Path
    task_description: str
    contract: Dict[str, Any]
    call_count: int = 0

    @property
    def calls_dir(self) -> Path:
        return self.path / "calls"

    def next_call_dir(self) -> Path:
        self.call_count += 1
        call_dir = self.calls_dir / f"{self.call_count:04d}"
        call_dir.mkdir(parents=True, exist_ok=False)
        return call_dir

    def write_evaluations(self, evaluations: Sequence[Any]) -> Path:
        path = self.path / "evaluations.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for evaluation in evaluations:
                handle.write(
                    json.dumps(evaluation.to_log_record(include_crn=True), sort_keys=True)
                    + "\n"
                )
        summary = {
            "candidate_count": len(evaluations),
            "valid_count": sum(bool(item.valid) for item in evaluations),
            "invalid_count": sum(not bool(item.valid) for item in evaluations),
            "best_loss": min(
                (
                    float(item.loss)
                    for item in evaluations
                    if item.valid and item.loss is not None
                ),
                default=None,
            ),
        }
        _write_json(self.path / "evaluation_summary.json", summary)
        return path


class HarnessLLMClient:
    """Provider adapter that invokes the Harness headless bot on demand.

    The model process receives only a dedicated run directory as its working
    directory. Prompts, raw responses, parsed JSON, and failures are retained in
    that directory. API keys remain in Harness configuration and are never
    copied into run artifacts.
    """

    def __init__(
        self,
        *,
        workspace_root: str | Path,
        dsh_home: str | Path,
        command: Optional[Sequence[str]] = None,
        timeout_seconds: float = 900.0,
        extra_environment: Optional[Mapping[str, str]] = None,
        system_prompt: str = CRN_AGENT_SYSTEM_PROMPT,
        provider: str = "deepseek-official",
        model: str = "deepseek-v4-flash",
        openai_compatible_base_url: Optional[str] = None,
        global_concurrency: int = 8,
        candidate_validation_policy: str = "atomic-batch",
        recover_valid_candidates: Optional[bool] = None,
    ):
        self.workspace_root = Path(workspace_root).expanduser().resolve()
        self.dsh_home = Path(dsh_home).expanduser().resolve()
        self.command = tuple(command or _default_dsh_command())
        self.timeout_seconds = float(timeout_seconds)
        self.extra_environment = dict(extra_environment or {})
        self.system_prompt = str(system_prompt).strip()
        self.provider = str(provider).strip()
        self.model = str(model).strip()
        self.openai_compatible_base_url = self._validate_local_base_url(
            openai_compatible_base_url
        )
        self.global_concurrency = int(global_concurrency)
        if recover_valid_candidates is not None:
            candidate_validation_policy = (
                "independent-members" if recover_valid_candidates else "atomic-batch"
            )
        self.candidate_validation_policy = str(candidate_validation_policy).strip().lower()
        if self.candidate_validation_policy not in {"atomic-batch", "independent-members"}:
            raise ValueError(
                "candidate_validation_policy must be 'atomic-batch' or "
                "'independent-members'."
            )
        self.recover_valid_candidates = self.candidate_validation_policy == "independent-members"
        if self.global_concurrency <= 0:
            raise ValueError("Harness global concurrency must be positive.")
        self.active_workspace: Optional[HarnessRunWorkspace] = None
        self.last_workspace: Optional[HarnessRunWorkspace] = None
        self.last_response_validation: Dict[str, Any] = {}

    def fork(self) -> "HarnessLLMClient":
        """Return an independent client for one concurrent Harness workspace."""

        return type(self)(
            workspace_root=self.workspace_root,
            dsh_home=self.dsh_home,
            command=self.command,
            timeout_seconds=self.timeout_seconds,
            extra_environment=self.extra_environment,
            system_prompt=self.system_prompt,
            provider=self.provider,
            model=self.model,
            openai_compatible_base_url=self.openai_compatible_base_url,
            global_concurrency=self.global_concurrency,
            candidate_validation_policy=self.candidate_validation_policy,
        )

    @contextmanager
    def run(
        self,
        *,
        task_description: str,
        contract: Optional[Mapping[str, Any]] = None,
        workspace_files: Optional[Mapping[str, Any]] = None,
        label: str = "crn-generation",
    ) -> Iterator[HarnessRunWorkspace]:
        """Create and activate one run workspace for one logical LLM round."""

        if self.active_workspace is not None:
            raise RuntimeError("A Harness run is already active on this client.")
        workspace = self._create_workspace(
            task_description=task_description,
            contract=dict(contract or {}),
            workspace_files=dict(workspace_files or {}),
            label=label,
        )
        self.active_workspace = workspace
        self.last_workspace = workspace
        try:
            yield workspace
        except Exception as exc:
            _write_json(
                workspace.path / "run_status.json",
                {"status": "failed", "error": f"{type(exc).__name__}: {exc}"},
            )
            raise
        else:
            _write_json(workspace.path / "run_status.json", {"status": "completed"})
        finally:
            self.active_workspace = None

    def generate_text(
        self,
        prompt: str,
        *,
        generation_config: Optional[LLMGenerationConfig] = None,
    ) -> str:
        """Run a text-generation call in a separate Harness process."""

        return self._invoke(prompt, generation_config=generation_config, response_kind="text")

    def generate_json(
        self,
        prompt: str,
        *,
        generation_config: Optional[LLMGenerationConfig] = None,
    ) -> Dict[str, Any]:
        """Run a JSON-generation call and strictly parse its response."""

        self.last_response_validation = {}
        raw = self._invoke(prompt, generation_config=generation_config, response_kind="json")
        payload = _loads_json_response(raw)
        if not isinstance(payload, dict):
            raise HarnessResponseError("Harness JSON response must be an object.")
        if self.active_workspace is not None and self.active_workspace.contract:
            payload, normalizations = _normalize_scalar_parameter_shorthand(
                payload, self.active_workspace.contract
            )
            if normalizations:
                _write_json(
                    self.active_workspace.path / "response_normalization.json",
                    {
                        "rule": "Normalize unambiguous scalar parameters to one vector per reaction.",
                        "candidates": normalizations,
                    },
                )
            payload, clamping = _clamp_crn_parameters(
                payload, self.active_workspace.contract
            )
            _write_json(
                self.active_workspace.path / "response_parameter_clamping.json",
                clamping,
            )
            if self.candidate_validation_policy == "independent-members":
                payload, validation = _recover_valid_crn_payload(
                    payload, self.active_workspace.contract
                )
                _write_json(
                    self.active_workspace.path / "response_member_validation.json",
                    validation,
                )
                if int(validation["accepted_candidate_count"]) == 0:
                    self.last_response_validation = {
                        **validation,
                        "clamped_parameter_count": int(
                            clamping["clamped_parameter_count"]
                        ),
                    }
                    raise HarnessResponseError(
                        "Harness returned no independently valid CRN candidates."
                    )
            else:
                _validate_crn_payload(payload, self.active_workspace.contract)
                validation = {
                    "policy": "atomic-batch",
                    "requested_candidate_count": int(
                        self.active_workspace.contract.get(
                            "required_candidate_count", len(payload.get("candidates", ()))
                        )
                    ),
                    "returned_candidate_count": len(payload.get("candidates", ())),
                    "accepted_candidate_count": len(payload.get("candidates", ())),
                    "accepted_candidate_indices": list(range(len(payload.get("candidates", ())))),
                    "rejected_candidates": [],
                }
                _write_json(
                    self.active_workspace.path / "response_member_validation.json",
                    validation,
                )
            self.last_response_validation = {
                **validation,
                "clamped_parameter_count": int(clamping["clamped_parameter_count"]),
            }
        if self.active_workspace is not None:
            _write_json(self.active_workspace.path / "latest_payload.json", payload)
        return payload

    def _create_workspace(
        self,
        *,
        task_description: str,
        contract: Dict[str, Any],
        workspace_files: Dict[str, Any],
        label: str,
    ) -> HarnessRunWorkspace:
        self.workspace_root.mkdir(parents=True, exist_ok=True)
        safe_label = "".join(char if char.isalnum() or char in "-_" else "-" for char in label)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        path = self.workspace_root / f"{stamp}-{safe_label}-{uuid4().hex[:8]}"
        path.mkdir(parents=False, exist_ok=False)
        (path / "calls").mkdir()
        workspace = HarnessRunWorkspace(path, task_description, contract)

        task_text = task_description.rstrip() + "\n"
        system_prompt_text = self.system_prompt + "\n"
        (path / "TASK.md").write_text(task_text, encoding="utf-8")
        (path / "SYSTEM_PROMPT.md").write_text(system_prompt_text, encoding="utf-8")
        profile_patch = path / "harness.patch.yml"
        profile_patch.write_text(self._profile_patch_text(), encoding="utf-8")
        _write_json(path / "OUTPUT_CONTRACT.json", contract)
        _write_json(path / "OUTPUT_GUIDE.json", _compact_output_guide(contract))
        (path / "REACTION_LIBRARY.tsv").write_text(
            _reaction_library_tsv(contract), encoding="utf-8"
        )
        for relative_name, content in workspace_files.items():
            relative_path = Path(relative_name)
            if relative_path.is_absolute() or ".." in relative_path.parts:
                raise ValueError(f"Workspace file must be a safe relative path: {relative_name!r}")
            destination = path / relative_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            if isinstance(content, bytes):
                destination.write_bytes(content)
            elif isinstance(content, str):
                destination.write_text(content.rstrip() + "\n", encoding="utf-8")
            else:
                _write_json(destination, content)
        _write_json(
            path / "run_manifest.json",
            {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "backend": "deepseek-harness-headless",
                "provider": self.provider,
                "model": self.model,
                "candidate_response_policy": self.candidate_validation_policy,
                "system_prompt_sha256": _sha256_text(system_prompt_text),
                "task_prompt_sha256": _sha256_text(task_text),
                "output_contract_sha256": _sha256_file(path / "OUTPUT_CONTRACT.json"),
                "output_guide_sha256": _sha256_file(path / "OUTPUT_GUIDE.json"),
                "reaction_library_sha256": _sha256_file(path / "REACTION_LIBRARY.tsv"),
                "workspace_file_sha256": {
                    name: _sha256_file(path / name) for name in sorted(workspace_files)
                },
                "dsh_home": str(self.dsh_home),
                "command": list(self.command),
                "security": {
                    "dedicated_working_directory": True,
                    "shell": False,
                    "secrets_written_to_workspace": False,
                },
            },
        )
        self._initialize_git(path)
        return workspace

    def _invoke(
        self,
        prompt: str,
        *,
        generation_config: Optional[LLMGenerationConfig],
        response_kind: str,
    ) -> str:
        if self.active_workspace is None:
            with self.run(task_description=prompt, label=f"standalone-{response_kind}"):
                return self._invoke(
                    prompt,
                    generation_config=generation_config,
                    response_kind=response_kind,
                )

        workspace = self.active_workspace
        call_dir = workspace.next_call_dir()
        config = generation_config or LLMGenerationConfig(
            response_mime_type="text/plain" if response_kind == "text" else "application/json"
        )
        request = (
            f"# Harness call {workspace.call_count}\n\n"
            f"Response kind: {response_kind}\n"
            f"Generation preferences: {json.dumps(config.to_dict(), sort_keys=True)}\n\n"
            "## Request\n\n"
            f"{prompt.rstrip()}\n"
        )
        request_path = call_dir / "request.md"
        request_path.write_text(request, encoding="utf-8")

        relative_request = request_path.relative_to(workspace.path)
        bot_task = build_harness_bot_task(response_kind, str(relative_request))
        profile_patch = workspace.path / "harness.patch.yml"

        queued_at = datetime.now(timezone.utc)
        queued = time.perf_counter()
        with self._global_request_slot() as slot:
            profile_name = self._worker_profile_name(slot)
            command = [
                *self.command,
                "--patch",
                str(profile_patch),
                "--profile",
                profile_name,
                bot_task,
            ]
            started_at = datetime.now(timezone.utc)
            started = time.perf_counter()
            queue_seconds = started - queued
            try:
                if self.openai_compatible_base_url is not None and response_kind == "json":
                    completed = self._run_local_json_handoff(
                        command, workspace=workspace, call_dir=call_dir
                    )
                else:
                    completed = subprocess.run(
                        command,
                        cwd=workspace.path,
                        env=self._child_environment(),
                        capture_output=True,
                        text=True,
                        timeout=self.timeout_seconds,
                        check=False,
                    )
            except (FileNotFoundError, PermissionError) as exc:
                raise HarnessUnavailableError(
                    f"Could not start DeepSeek Harness with {self.command!r}: {exc}"
                ) from exc
            except subprocess.TimeoutExpired as exc:
                _write_json(
                    call_dir / "process.json",
                    {
                        "queued_at": queued_at.isoformat(),
                        "queue_seconds": queue_seconds,
                        "started_at": started_at.isoformat(),
                        "duration_seconds": time.perf_counter() - started,
                        "status": "timeout",
                        "slot": slot,
                        "response_kind": response_kind,
                        "provider": self.provider,
                        "model": self.model,
                        "profile": profile_name,
                    },
                )
                raise HarnessUnavailableError(
                    f"DeepSeek Harness exceeded the {self.timeout_seconds:g}s timeout."
                ) from exc

        (call_dir / "stdout.txt").write_text(completed.stdout, encoding="utf-8")
        (call_dir / "stderr.txt").write_text(completed.stderr, encoding="utf-8")
        _write_json(
            call_dir / "process.json",
            {
                "started_at": started_at.isoformat(),
                "queued_at": queued_at.isoformat(),
                "queue_seconds": queue_seconds,
                "duration_seconds": time.perf_counter() - started,
                "slot": slot,
                "returncode": completed.returncode,
                "response_kind": response_kind,
                "provider": self.provider,
                "model": self.model,
                "profile": profile_name,
            },
        )
        if completed.returncode != 0:
            raise HarnessUnavailableError(
                "DeepSeek Harness failed; inspect "
                f"{call_dir / 'stderr.txt'} (exit code {completed.returncode})."
            )
        if not completed.stdout.strip():
            raise HarnessResponseError("DeepSeek Harness returned an empty response.")
        return completed.stdout.strip()

    def _run_local_json_handoff(
        self,
        command: Sequence[str],
        *,
        workspace: HarnessRunWorkspace,
        call_dir: Path,
    ) -> subprocess.CompletedProcess[str]:
        """Accept a validated workspace handoff without waiting on a looping agent."""

        process = subprocess.Popen(
            command,
            cwd=workspace.path,
            env=self._child_environment(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        response_path = workspace.path / "FINAL_RESPONSE.json"
        error_path = workspace.path / "FINAL_RESPONSE.validation-error.txt"
        deadline = time.monotonic() + self.timeout_seconds
        observed_signature: Optional[tuple[int, int]] = None

        while process.poll() is None:
            if response_path.is_file():
                stat = response_path.stat()
                signature = (stat.st_mtime_ns, stat.st_size)
                if signature != observed_signature:
                    observed_signature = signature
                    try:
                        payload = json.loads(response_path.read_text(encoding="utf-8"))
                        payload, _ = _normalize_scalar_parameter_shorthand(
                            payload, workspace.contract
                        )
                        if self.recover_valid_candidates:
                            payload, _ = _recover_valid_crn_payload(
                                payload, workspace.contract
                            )
                        else:
                            _validate_crn_payload(payload, workspace.contract)
                    except (OSError, json.JSONDecodeError, HarnessResponseError) as exc:
                        error_path.write_text(f"{type(exc).__name__}: {exc}\n", encoding="utf-8")
                    else:
                        process.terminate()
                        try:
                            _, stderr = process.communicate(timeout=10)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            _, stderr = process.communicate()
                        _write_json(
                            call_dir / "workspace_handoff.json",
                            {"path": "FINAL_RESPONSE.json", "validated": True},
                        )
                        return subprocess.CompletedProcess(
                            command,
                            0,
                            stdout=json.dumps(payload),
                            stderr=stderr,
                        )
            if time.monotonic() >= deadline:
                process.terminate()
                try:
                    stdout, stderr = process.communicate(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    stdout, stderr = process.communicate()
                raise subprocess.TimeoutExpired(
                    command, self.timeout_seconds, output=stdout, stderr=stderr
                )
            time.sleep(0.25)

        stdout, stderr = process.communicate()
        return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)

    @contextmanager
    def _global_request_slot(self) -> Iterator[int]:
        """Bound Harness processes across independently launched seed workers."""

        pool_name = "".join(
            character if character.isalnum() or character in "-_" else "-"
            for character in self.provider
        ) or "default"
        slot_root = self.dsh_home / "request-slots" / pool_name
        slot_root.mkdir(parents=True, exist_ok=True)
        handles = [
            (slot_root / f"slot-{index:03d}.lock").open("a+", encoding="utf-8")
            for index in range(self.global_concurrency)
        ]
        acquired = None
        try:
            while acquired is None:
                for index, handle in enumerate(handles):
                    try:
                        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        continue
                    acquired = index
                    break
                if acquired is None:
                    time.sleep(0.25)
            yield acquired
        finally:
            if acquired is not None:
                fcntl.flock(handles[acquired].fileno(), fcntl.LOCK_UN)
            for handle in handles:
                handle.close()

    def _child_environment(self) -> Dict[str, str]:
        allowed = (
            "HOME",
            "USER",
            "LOGNAME",
            "SHELL",
            "PATH",
            "LANG",
            "LC_ALL",
            "TMPDIR",
            "XDG_RUNTIME_DIR",
            "SSL_CERT_FILE",
            "SSL_CERT_DIR",
            "HTTP_PROXY",
            "HTTPS_PROXY",
            "NO_PROXY",
        )
        child = {key: os.environ[key] for key in allowed if key in os.environ}
        command_path = Path(self.command[0])
        if command_path.is_absolute():
            current_path = child.get("PATH", os.defpath)
            child["PATH"] = f"{command_path.parent}{os.pathsep}{current_path}"
        child["DSH_HOME"] = str(self.dsh_home)
        if self.openai_compatible_base_url is not None:
            # pi-ai requires an API-key-shaped value even when llama.cpp's
            # loopback endpoint is intentionally configured without auth.
            child["DSH_LOCAL_LLM_API_KEY"] = "local-loopback-no-auth"
        child.update({str(key): str(value) for key, value in self.extra_environment.items()})
        return child

    def _worker_profile_name(self, slot: int) -> str:
        """Create one reusable, exclusively locked DSH profile per global slot."""

        template = self.dsh_home / "profiles" / "headless"
        if not template.is_dir():
            return "headless"

        pool_name = "".join(
            character if character.isalnum() or character in "-_" else "-"
            for character in self.provider
        ) or "default"
        name = f"headless-worker-{pool_name}-{slot:03d}"
        destination = self.dsh_home / "profiles" / name
        if destination.is_dir():
            return name

        destination.mkdir(parents=False, exist_ok=False)
        for filename in ("cordis.patch.yml", "pnpm-workspace.yaml"):
            shutil.copy2(template / filename, destination / filename)

        package = json.loads((template / "package.json").read_text(encoding="utf-8"))
        package["name"] = f"dsh-profile-{name}"
        _write_json(destination / "package.json", package)
        (destination / "cordis.yml").write_text("[]\n", encoding="utf-8")
        return name

    def _profile_patch_text(self) -> str:
        persona = "\n".join(f"      {line}" for line in self.system_prompt.splitlines())
        local_provider = ""
        if self.openai_compatible_base_url is not None:
            local_provider = (
                "- id: llm-pi-ai\n"
                "  config:\n"
                "    providers:\n"
                f"      {json.dumps(self.provider)}:\n"
                "        apiKeyEnv: DSH_LOCAL_LLM_API_KEY\n"
                "        displayName: Local llama.cpp\n"
                "        api: openai-completions\n"
                f"        baseURL: {json.dumps(self.openai_compatible_base_url)}\n"
                "        compat:\n"
                "          supportsDeveloperRole: false\n"
                "          maxTokensField: max_tokens\n"
                "        models:\n"
                f"          - id: {json.dumps(self.model)}\n"
                f"            name: {json.dumps(self.model)}\n"
                "            contextWindow: 32768\n"
                "            maxTokens: 4096\n"
            )
        return local_provider + (
            "- id: agent-default-model\n"
            "  config:\n"
            f"    provider: {json.dumps(self.provider)}\n"
            f"    model: {json.dumps(self.model)}\n"
            "- id: system-prompt\n"
            "  config:\n"
            "    persona: |-\n"
            f"{persona}\n"
        )

    @staticmethod
    def _validate_local_base_url(value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        base_url = str(value).strip().rstrip("/")
        parsed = urlparse(base_url)
        if parsed.scheme not in {"http", "https"} or parsed.hostname not in {
            "127.0.0.1",
            "localhost",
            "::1",
        }:
            raise ValueError(
                "The OpenAI-compatible Harness endpoint must be an HTTP(S) localhost URL."
            )
        return base_url

    @staticmethod
    def _initialize_git(path: Path) -> None:
        if shutil.which("git") is None:
            return
        initialized = subprocess.run(
            ["git", "init", "--quiet"],
            cwd=path,
            capture_output=True,
            text=True,
            check=False,
        )
        if initialized.returncode != 0:
            return
        subprocess.run(
            ["git", "add", "--all"],
            cwd=path,
            capture_output=True,
            text=True,
            check=False,
        )
        subprocess.run(
            [
                "git",
                "-c",
                "user.name=RL4CRN Harness",
                "-c",
                "user.email=harness@localhost",
                "commit",
                "--quiet",
                "-m",
                "Initialize CRN Harness run",
            ],
            cwd=path,
            capture_output=True,
            text=True,
            check=False,
        )


def build_crn_output_contract(
    evaluator: Any,
    *,
    num_candidates: int,
    task_description: str,
    candidate_validation_policy: str = "independent-members",
) -> Dict[str, Any]:
    """Build a task-specific output contract from an active evaluator."""

    reactions = []
    forbidden_reaction_ids = set(getattr(evaluator, "forbidden_reaction_ids", ()))
    template_reaction_ids = sorted(
        int(value) for value in getattr(evaluator, "initial_reaction_ids", ())
    )
    try:
        null_reaction_id = int(evaluator.library.find_zero_reaction())
    except (AttributeError, TypeError, ValueError):
        null_reaction_id = None
    for reaction_id in range(len(evaluator.library)):
        if reaction_id in forbidden_reaction_ids:
            continue
        reaction = evaluator.library.get_reaction(reaction_id)
        reactions.append(
            {
                "id": reaction_id,
                "parameter_count": int(getattr(reaction, "num_parameters", 0)),
                "display": str(reaction),
            }
        )
    budget = int(evaluator.max_added_reactions)
    allowed_reaction_ids = [item["id"] for item in reactions]
    enforce_bounds = bool(getattr(evaluator, "enforce_parameter_bounds", False))
    parameter_minimum = float(getattr(evaluator, "min_parameter_value", 1e-6))
    parameter_maximum = getattr(evaluator, "max_parameter_value", None)
    allowed_range = None
    if enforce_bounds:
        allowed_range = [
            parameter_minimum,
            float(parameter_maximum) if parameter_maximum is not None else None,
        ]
    parameter_schema: Dict[str, Any] = {"type": "number", "exclusiveMinimum": 0}
    if enforce_bounds:
        parameter_schema = {"type": "number", "minimum": parameter_minimum}
        if parameter_maximum is not None:
            parameter_schema["maximum"] = float(parameter_maximum)
    return {
        "contract_version": 2,
        "proposal_space_contract_version": 2,
        "task": task_description,
        "template_reaction_ids": template_reaction_ids,
        "null_reaction_id": null_reaction_id,
        "forbidden_reaction_ids": sorted(forbidden_reaction_ids),
        "required_candidate_count": int(num_candidates),
        "rules": {
            "reaction_count_per_candidate": budget,
            "reaction_ids_must_be_unique": bool(evaluator.require_unique_reactions),
            "reaction_ids_must_exist_in_library": True,
            "forbidden_reaction_ids": sorted(forbidden_reaction_ids),
            "parameters_must_be_positive_finite_numbers": True,
            "allowed_parameter_range": allowed_range,
            "out_of_range_parameter_policy": "clamp" if enforce_bounds else "reject",
            "candidate_validation_policy": str(candidate_validation_policy),
            "parameter_vector_length_must_match_reaction": True,
            "no_markdown_or_text_outside_json": True,
            "candidates_must_not_be_exact_duplicates": True,
        },
        "reaction_library": reactions,
        "json_schema": {
            "type": "object",
            "required": ["candidates"],
            "additionalProperties": False,
            "properties": {
                "candidates": {
                    "type": "array",
                    "minItems": int(num_candidates),
                    "maxItems": int(num_candidates),
                    "items": {
                        "type": "object",
                        "required": ["reaction_ids", "parameter_values"],
                        "additionalProperties": False,
                        "properties": {
                            "reaction_ids": {
                                "type": "array",
                                "minItems": budget,
                                "maxItems": budget,
                                "uniqueItems": bool(evaluator.require_unique_reactions),
                                "items": {"type": "integer", "enum": allowed_reaction_ids},
                            },
                            "parameter_values": {
                                "type": "array",
                                "minItems": budget,
                                "maxItems": budget,
                                "items": {
                                    "type": "array",
                                    "items": parameter_schema,
                                },
                            },
                        },
                    },
                }
            },
        },
    }


def _default_dsh_command() -> Sequence[str]:
    configured = os.environ.get("RL4CRN_DSH_COMMAND")
    if configured:
        return shlex.split(configured)
    direct = shutil.which("dsh")
    if direct:
        return [direct]
    installed = _installed_dsh_command()
    if installed is not None:
        return installed
    corepack = shutil.which("corepack")
    if corepack:
        return [corepack, "pnpm", "dlx", DEFAULT_DSH_PACKAGE]
    user_corepack = Path.home() / ".local" / "nodejs" / "current" / "bin" / "corepack"
    if user_corepack.is_file():
        return [str(user_corepack), "pnpm", "dlx", DEFAULT_DSH_PACKAGE]
    return ["npx", "--yes", DEFAULT_DSH_PACKAGE]


def _installed_dsh_command() -> Optional[list[str]]:
    """Resolve the pinned workspace installation without running a package manager."""

    version = DEFAULT_DSH_PACKAGE.rsplit("@", 1)[-1]
    node = shutil.which("node")
    if node is None:
        user_node = Path.home() / ".local" / "nodejs" / "current" / "bin" / "node"
        if user_node.is_file():
            node = str(user_node)
    if node is None:
        return None
    package_root = (
        Path.home()
        / "ai-workspaces/deepseek-test/dsh-runtime/node_modules/@deepseek-ai/dsh"
    )
    executable = package_root / "lib/bin.js"
    manifest = package_root / "package.json"
    if not executable.is_file() or not manifest.is_file():
        return None
    try:
        installed_version = json.loads(manifest.read_text(encoding="utf-8"))["version"]
    except (OSError, KeyError, json.JSONDecodeError):
        return None
    if installed_version != version:
        return None
    return [node, str(executable)]


def _loads_json_response(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError as first_error:
        decoder = json.JSONDecoder()
        # CRN responses must be objects. Prefer object starts so prose such as
        # "parameters in [0.1, 50.0]" cannot be mistaken for the payload.
        for opening in ("{", "["):
            for index, char in enumerate(text):
                if char != opening:
                    continue
                try:
                    value, _ = decoder.raw_decode(text[index:])
                    return value
                except json.JSONDecodeError:
                    continue
        raise HarnessResponseError(f"Harness returned invalid JSON: {first_error}") from first_error


def _validate_crn_payload(payload: Mapping[str, Any], contract: Mapping[str, Any]) -> None:
    """Enforce the task-specific candidate shape before simulation."""

    if set(payload) != {"candidates"}:
        raise HarnessResponseError("CRN output must contain only the 'candidates' field.")
    candidates = payload.get("candidates")
    if not isinstance(candidates, list):
        raise HarnessResponseError("The 'candidates' field must be a list.")

    required_count = int(contract.get("required_candidate_count", len(candidates)))
    if len(candidates) != required_count:
        raise HarnessResponseError(
            f"Harness returned {len(candidates)} candidates; expected {required_count}."
        )

    rules = dict(contract.get("rules", {}))
    budget = int(rules.get("reaction_count_per_candidate", 0))
    reaction_specs = {
        int(item["id"]): int(item["parameter_count"])
        for item in contract.get("reaction_library", [])
    }
    allowed_range = rules.get("allowed_parameter_range")
    candidate_signatures = set()
    for candidate_index, candidate in enumerate(candidates):
        if not isinstance(candidate, Mapping):
            raise HarnessResponseError(f"Candidate {candidate_index} must be an object.")
        if set(candidate) != {"reaction_ids", "parameter_values"}:
            raise HarnessResponseError(
                f"Candidate {candidate_index} must contain only reaction_ids and parameter_values."
            )
        reaction_ids = candidate.get("reaction_ids")
        parameter_values = candidate.get("parameter_values")
        if not isinstance(reaction_ids, list) or not isinstance(parameter_values, list):
            raise HarnessResponseError(
                f"Candidate {candidate_index} reaction_ids and parameter_values must be lists."
            )
        if len(reaction_ids) != budget or len(parameter_values) != budget:
            raise HarnessResponseError(
                f"Candidate {candidate_index} must contain exactly {budget} reactions and parameter vectors."
            )
        for position, reaction_id in enumerate(reaction_ids):
            if isinstance(reaction_id, bool) or not isinstance(reaction_id, int):
                raise HarnessResponseError(
                    f"Candidate {candidate_index} reaction ID at position {position} must be an integer."
                )
        if rules.get("reaction_ids_must_be_unique", True) and len(set(reaction_ids)) != len(reaction_ids):
            raise HarnessResponseError(f"Candidate {candidate_index} contains duplicate reaction IDs.")

        for position, (reaction_id, parameters) in enumerate(zip(reaction_ids, parameter_values)):
            if reaction_id not in reaction_specs:
                raise HarnessResponseError(
                    f"Candidate {candidate_index} uses unknown reaction ID {reaction_id}."
                )
            if not isinstance(parameters, list):
                raise HarnessResponseError(
                    f"Candidate {candidate_index} parameters at position {position} must be a list."
                )
            expected = reaction_specs[reaction_id]
            if len(parameters) != expected:
                raise HarnessResponseError(
                    f"Candidate {candidate_index} reaction ID {reaction_id} expects {expected} parameters; "
                    f"got {len(parameters)}."
                )
            for value in parameters:
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise HarnessResponseError(
                        f"Candidate {candidate_index} parameters must be numbers."
                    )
                if not float("-inf") < float(value) < float("inf") or float(value) <= 0:
                    raise HarnessResponseError(
                        f"Candidate {candidate_index} parameters must be positive finite numbers."
                    )
                if allowed_range is not None:
                    lower, upper = allowed_range
                    if float(value) < float(lower) or (
                        upper is not None and float(value) > float(upper)
                    ):
                        raise HarnessResponseError(
                            f"Candidate {candidate_index} parameters must be within "
                            f"[{lower}, {upper}]."
                        )
        signature = tuple(
            sorted(
                (int(reaction_id), tuple(float(value) for value in parameters))
                for reaction_id, parameters in zip(reaction_ids, parameter_values)
            )
        )
        if signature in candidate_signatures:
            raise HarnessResponseError(
                f"Candidate {candidate_index} exactly duplicates an earlier candidate."
            )
        candidate_signatures.add(signature)


def _recover_valid_crn_payload(
    payload: Mapping[str, Any], contract: Mapping[str, Any]
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """Retain independently valid candidates and audit every rejected member."""

    raw_candidates = payload.get("candidates")
    if not isinstance(raw_candidates, list):
        raise HarnessResponseError("The 'candidates' field must be a list.")

    required_count = int(contract.get("required_candidate_count", len(raw_candidates)))
    accepted: list[Dict[str, Any]] = []
    accepted_indices = []
    rejected = []
    projections = []
    for candidate_index, raw_candidate in enumerate(raw_candidates):
        if len(accepted) >= required_count:
            rejected.append(
                {
                    "candidate_index": candidate_index,
                    "error": "Surplus candidate beyond the requested batch size.",
                }
            )
            continue
        if not isinstance(raw_candidate, Mapping):
            rejected.append(
                {
                    "candidate_index": candidate_index,
                    "error": "Candidate must be an object.",
                }
            )
            continue

        candidate = {
            key: raw_candidate[key]
            for key in ("reaction_ids", "parameter_values")
            if key in raw_candidate
        }
        ignored_fields = sorted(set(raw_candidate) - set(candidate))
        if ignored_fields:
            projections.append(
                {
                    "candidate_index": candidate_index,
                    "ignored_annotation_fields": ignored_fields,
                }
            )

        candidate_contract = dict(contract)
        candidate_contract["required_candidate_count"] = len(accepted) + 1
        try:
            _validate_crn_payload(
                {"candidates": [*accepted, candidate]}, candidate_contract
            )
        except HarnessResponseError as exc:
            rejected.append(
                {"candidate_index": candidate_index, "error": str(exc)}
            )
            continue
        accepted.append(candidate)
        accepted_indices.append(candidate_index)

    recovery = {
        "policy": "independent-members",
        "requested_candidate_count": required_count,
        "returned_candidate_count": len(raw_candidates),
        "accepted_candidate_count": len(accepted),
        "accepted_candidate_indices": accepted_indices,
        "rejected_candidates": rejected,
        "field_projections": projections,
        "ignored_top_level_fields": sorted(set(payload) - {"candidates"}),
        "scientific_values_modified": False,
    }
    return {"candidates": accepted}, recovery


def _clamp_crn_parameters(
    payload: Mapping[str, Any], contract: Mapping[str, Any]
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """Clamp finite numeric rates when the contract declares endpoint truncation."""

    normalized = dict(payload)
    rules = dict(contract.get("rules", {}))
    allowed_range = rules.get("allowed_parameter_range")
    policy = rules.get("out_of_range_parameter_policy", "reject")
    raw_candidates = payload.get("candidates")
    if policy != "clamp" or allowed_range is None or not isinstance(raw_candidates, list):
        return normalized, {
            "policy": str(policy),
            "allowed_parameter_range": allowed_range,
            "clamped_parameter_count": 0,
            "clamps": [],
        }

    lower, upper = allowed_range
    lower = float(lower)
    upper = None if upper is None else float(upper)
    candidates = []
    clamps: list[Dict[str, Any]] = []
    for candidate_index, raw_candidate in enumerate(raw_candidates):
        if not isinstance(raw_candidate, Mapping):
            candidates.append(raw_candidate)
            continue
        candidate = dict(raw_candidate)
        raw_vectors = candidate.get("parameter_values")
        if not isinstance(raw_vectors, list):
            candidates.append(candidate)
            continue
        vectors = []
        for reaction_position, raw_vector in enumerate(raw_vectors):
            if not isinstance(raw_vector, list):
                vectors.append(raw_vector)
                continue
            vector = []
            for parameter_position, raw_value in enumerate(raw_vector):
                if (
                    isinstance(raw_value, bool)
                    or not isinstance(raw_value, (int, float))
                    or not math.isfinite(float(raw_value))
                ):
                    vector.append(raw_value)
                    continue
                value = float(raw_value)
                clamped = max(lower, value)
                if upper is not None:
                    clamped = min(upper, clamped)
                if clamped != value:
                    clamps.append(
                        {
                            "candidate_index": candidate_index,
                            "reaction_position": reaction_position,
                            "parameter_position": parameter_position,
                            "original_value": raw_value,
                            "clamped_value": clamped,
                        }
                    )
                vector.append(clamped)
            vectors.append(vector)
        candidate["parameter_values"] = vectors
        candidates.append(candidate)
    normalized["candidates"] = candidates
    return normalized, {
        "policy": "clamp",
        "allowed_parameter_range": [lower, upper],
        "clamped_parameter_count": len(clamps),
        "clamps": clamps,
    }


def _normalize_scalar_parameter_shorthand(
    payload: Mapping[str, Any], contract: Mapping[str, Any]
) -> tuple[Dict[str, Any], list[Dict[str, Any]]]:
    """Normalize the two unambiguous scalar-rate shapes emitted by Harness.

    Raw model output remains in ``calls/*/stdout.txt``. This only accepts a
    shorthand when every selected reaction has exactly one parameter.
    """

    normalized = dict(payload)
    raw_candidates = payload.get("candidates")
    if not isinstance(raw_candidates, list):
        return normalized, []
    reaction_specs = {
        int(item["id"]): int(item["parameter_count"])
        for item in contract.get("reaction_library", [])
    }
    candidates = []
    changes: list[Dict[str, Any]] = []
    for candidate_index, raw_candidate in enumerate(raw_candidates):
        if not isinstance(raw_candidate, Mapping):
            candidates.append(raw_candidate)
            continue
        candidate = dict(raw_candidate)
        reaction_ids = candidate.get("reaction_ids")
        parameters = candidate.get("parameter_values")
        if not isinstance(reaction_ids, list) or not isinstance(parameters, list):
            candidates.append(candidate)
            continue
        if not reaction_ids or any(reaction_specs.get(reaction_id) != 1 for reaction_id in reaction_ids):
            candidates.append(candidate)
            continue

        shape = None
        scalar_values = None
        if len(parameters) == len(reaction_ids) and all(
            isinstance(value, (int, float)) and not isinstance(value, bool)
            for value in parameters
        ):
            shape = "flat_scalars"
            scalar_values = parameters
        elif (
            len(parameters) == 1
            and isinstance(parameters[0], list)
            and len(parameters[0]) == len(reaction_ids)
            and all(
                isinstance(value, (int, float)) and not isinstance(value, bool)
                for value in parameters[0]
            )
        ):
            shape = "single_packed_vector"
            scalar_values = parameters[0]

        if scalar_values is not None:
            candidate["parameter_values"] = [[value] for value in scalar_values]
            changes.append({"candidate_index": candidate_index, "input_shape": shape})
        candidates.append(candidate)
    normalized["candidates"] = candidates
    return normalized, changes


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _compact_output_guide(contract: Mapping[str, Any]) -> Dict[str, Any]:
    """Expose the machine contract without embedding its large reaction catalog."""

    return {
        "contract_version": contract.get("contract_version"),
        "required_candidate_count": contract.get("required_candidate_count"),
        "output_shape": {
            "candidates": [
                {
                    "reaction_ids": ["integer reaction ID"],
                    "parameter_values": [["one numeric vector per reaction"]],
                }
            ]
        },
        "rules": contract.get("rules", {}),
        "reaction_library": {
            "path": "REACTION_LIBRARY.tsv",
            "columns": ["id", "parameter_count", "display"],
            "lookup": "Use targeted grep queries; do not read the full file.",
        },
        "authoritative_contract": "OUTPUT_CONTRACT.json",
    }


def _reaction_library_tsv(contract: Mapping[str, Any]) -> str:
    lines = ["id\tparameter_count\tdisplay"]
    for entry in contract.get("reaction_library", []):
        display = str(entry.get("display", "")).replace("\t", " ").replace("\n", " ")
        lines.append(
            f"{entry['id']}\t{entry['parameter_count']}\t{display}"
        )
    return "\n".join(lines) + "\n"


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
