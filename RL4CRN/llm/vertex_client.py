"""Optional VertexAI/Gemini client for LLM-assisted CRN generation."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional

from RL4CRN.llm.schemas import LLMGenerationConfig


DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
DEFAULT_GEMINI_FALLBACK_MODELS = ("gemini-2.5-flash", "gemini-2.5-flash-lite")


class VertexAIUnavailableError(ImportError):
    """Raised when VertexAI support is requested but not installed."""


class GeminiAPIKeyLLMClient:
    """Gemini client using a Google AI Studio / Gemini API key.

    This backend does not use VertexAI projects.  It is useful when the user has
    an API key but not a billable Google Cloud project with Vertex AI enabled.
    """

    def __init__(
        self,
        *,
        api_key: str,
        model_name: str = DEFAULT_GEMINI_MODEL,
        fallback_model_names: tuple[str, ...] = DEFAULT_GEMINI_FALLBACK_MODELS,
    ):
        if not api_key:
            raise ValueError(
                "A Gemini API key is required. Set GEMINI_API_KEY or GOOGLE_API_KEY "
                "before constructing GeminiAPIKeyLLMClient."
            )
        self.api_key = api_key
        self.model_name = model_name
        self.fallback_model_names = tuple(
            model for model in fallback_model_names if model and model != model_name
        )

        try:
            from google import genai
        except ImportError as exc:
            raise VertexAIUnavailableError(
                "Gemini API-key support requires the Google Gen AI SDK. "
                "Install RL4CRN with the 'vertexai' extra, or install 'google-genai'."
            ) from exc

        self._client = genai.Client(api_key=api_key)

    def generate_json(
        self,
        prompt: str,
        *,
        generation_config: Optional[LLMGenerationConfig] = None,
    ) -> Dict[str, Any]:
        """Generate JSON from a prompt and parse the model response."""

        cfg_dict = _json_generation_config(generation_config)
        response = self._generate_content(
            contents=prompt,
            config=cfg_dict,
        )
        text = getattr(response, "text", "")
        return _loads_model_json(
            text,
            repair_fn=lambda bad_text, error: getattr(
                self._generate_content(
                    contents=_build_json_repair_prompt(bad_text, error),
                    config={**cfg_dict, "temperature": 0.0},
                ),
                "text",
                "",
            ),
        )

    def generate_text(
        self,
        prompt: str,
        *,
        generation_config: Optional[LLMGenerationConfig] = None,
    ) -> str:
        """Generate free text for graph roles such as deciders and critics."""

        cfg = generation_config or LLMGenerationConfig(response_mime_type="text/plain")
        cfg_dict = cfg.to_dict()
        cfg_dict.pop("response_mime_type", None)
        response = self._generate_content(
            contents=prompt,
            config=cfg_dict,
        )
        return str(getattr(response, "text", ""))

    def _generate_content(self, *, contents: str, config: Dict[str, Any]) -> Any:
        models_to_try = (self.model_name, *self.fallback_model_names)
        last_exc = None
        for model_name in models_to_try:
            try:
                response = self._client.models.generate_content(
                    model=model_name,
                    contents=contents,
                    config=config,
                )
                self.model_name = model_name
                return response
            except Exception as exc:
                if not _is_model_not_found_error(exc):
                    raise
                last_exc = exc
        raise last_exc


class VertexLLMClient:
    """Thin wrapper around Gemini models on Google Cloud.

    The import of Google Cloud dependencies is intentionally lazy.  This keeps
    RL4CRN usable on machines without Google Cloud dependencies or billing
    access; the client only fails when a Vertex model is actually constructed.

    The preferred backend is the Google Gen AI SDK.  If that package is not
    installed, the wrapper falls back to the legacy ``vertexai.generative_models``
    interface for older environments.
    """

    def __init__(
        self,
        *,
        project_id: str,
        location: str = "europe-west1",
        model_name: str = DEFAULT_GEMINI_MODEL,
        fallback_model_names: tuple[str, ...] = DEFAULT_GEMINI_FALLBACK_MODELS,
    ):
        if not project_id or project_id == "YOUR_GOOGLE_CLOUD_PROJECT":
            raise ValueError(
                "A real Google Cloud project id is required for VertexAI. "
                "Set GOOGLE_CLOUD_PROJECT in the environment before constructing VertexLLMClient."
            )
        self.project_id = project_id
        self.location = location
        self.model_name = model_name
        self.fallback_model_names = tuple(
            model for model in fallback_model_names if model and model != model_name
        )
        self._backend = "genai"

        try:
            from google import genai

            self._client = genai.Client(
                vertexai=True,
                project=project_id,
                location=location,
            )
            self._model = None
        except ImportError:
            self._backend = "legacy_vertexai"
            try:
                import vertexai
                from vertexai.generative_models import GenerativeModel
            except ImportError as exc:
                raise VertexAIUnavailableError(
                    "VertexAI support requires Google Cloud generation dependencies. "
                    "Install RL4CRN with the 'vertexai' extra, or install "
                    "'google-genai' and 'google-cloud-aiplatform' in this environment."
                ) from exc

            vertexai.init(project=project_id, location=location)
            self._client = None
            self._model = GenerativeModel(model_name)

    def generate_json(
        self,
        prompt: str,
        *,
        generation_config: Optional[LLMGenerationConfig] = None,
    ) -> Dict[str, Any]:
        """Generate JSON from a prompt and parse the model response."""

        cfg_dict = _json_generation_config(generation_config)

        if self._backend == "genai":
            response = self._generate_content_genai(
                contents=prompt,
                config=cfg_dict,
            )
            text = getattr(response, "text", "")
            return _loads_model_json(
                text,
                repair_fn=lambda bad_text, error: getattr(
                    self._generate_content_genai(
                        contents=_build_json_repair_prompt(bad_text, error),
                        config={**cfg_dict, "temperature": 0.0},
                    ),
                    "text",
                    "",
                ),
            )

        try:
            from vertexai.generative_models import GenerationConfig
        except ImportError as exc:
            raise VertexAIUnavailableError("VertexAI generation classes are unavailable.") from exc

        response = self._model.generate_content(prompt, generation_config=GenerationConfig(**cfg_dict))
        text = getattr(response, "text", "")
        return _loads_model_json(text)

    def generate_text(
        self,
        prompt: str,
        *,
        generation_config: Optional[LLMGenerationConfig] = None,
    ) -> str:
        """Generate free text for graph roles such as deciders and critics."""

        cfg = generation_config or LLMGenerationConfig(response_mime_type="text/plain")
        cfg_dict = cfg.to_dict()
        cfg_dict.pop("response_mime_type", None)

        if self._backend == "genai":
            response = self._generate_content_genai(
                contents=prompt,
                config=cfg_dict,
            )
            return str(getattr(response, "text", ""))

        try:
            from vertexai.generative_models import GenerationConfig
        except ImportError as exc:
            raise VertexAIUnavailableError("VertexAI generation classes are unavailable.") from exc

        response = self._model.generate_content(prompt, generation_config=GenerationConfig(**cfg_dict))
        return str(getattr(response, "text", ""))

    def _generate_content_genai(self, *, contents: str, config: Dict[str, Any]) -> Any:
        models_to_try = (self.model_name, *self.fallback_model_names)
        last_exc = None
        for model_name in models_to_try:
            try:
                response = self._client.models.generate_content(
                    model=model_name,
                    contents=contents,
                    config=config,
                )
                self.model_name = model_name
                return response
            except Exception as exc:
                if not _is_model_not_found_error(exc):
                    raise
                last_exc = exc
        raise last_exc


def _is_model_not_found_error(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    if status_code == 404:
        return True
    return "NOT_FOUND" in str(exc) and "models/" in str(exc)


def _json_generation_config(generation_config: Optional[LLMGenerationConfig]) -> Dict[str, Any]:
    cfg = generation_config or LLMGenerationConfig()
    cfg_dict = cfg.to_dict()
    cfg_dict.setdefault("response_mime_type", "application/json")
    cfg_dict.setdefault("max_output_tokens", 16384)
    return cfg_dict


def _loads_model_json(text: str, *, repair_fn: Optional[Any] = None) -> Dict[str, Any]:
    """Parse JSON returned by a model, with one optional repair attempt.

    Even with ``response_mime_type='application/json'``, preview or overloaded
    models can occasionally emit markdown fences, explanatory text, or truncated
    JSON.  We first try strict parsing, then extract the first balanced JSON
    object/array.  If that still fails and a repair callable is provided, the
    model gets one deterministic chance to rewrite the payload as valid JSON.
    """

    if text is None:
        text = ""
    text = str(text)

    try:
        return json.loads(text)
    except json.JSONDecodeError as first_error:
        extracted = _extract_json_block(text)
        if extracted and extracted != text:
            try:
                return json.loads(extracted)
            except json.JSONDecodeError as extracted_error:
                first_error = extracted_error

        if repair_fn is not None:
            repaired = str(repair_fn(text, first_error) or "")
            try:
                return json.loads(repaired)
            except json.JSONDecodeError:
                extracted = _extract_json_block(repaired)
                if extracted:
                    try:
                        return json.loads(extracted)
                    except json.JSONDecodeError:
                        pass

        preview = text[:1000].replace("\n", "\\n")
        raise ValueError(
            "Gemini returned invalid JSON and automatic repair failed. "
            f"Original parse error: {first_error}. Response preview: {preview!r}"
        ) from first_error


def _extract_json_block(text: str) -> Optional[str]:
    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        text = fenced.group(1)

    starts = [idx for idx in (text.find("{"), text.find("[")) if idx >= 0]
    if not starts:
        return None
    start = min(starts)

    stack = []
    in_string = False
    escaped = False
    for idx in range(start, len(text)):
        ch = text[idx]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch in "{[":
            stack.append("}" if ch == "{" else "]")
        elif ch in "}]":
            if not stack or ch != stack[-1]:
                return None
            stack.pop()
            if not stack:
                return text[start : idx + 1]
    return None


def _build_json_repair_prompt(bad_text: str, error: Exception) -> str:
    return (
        "Repair the following invalid JSON so that it is valid JSON only. "
        "Keep the same schema and content as much as possible. Do not add markdown, "
        "comments, or explanations.\n\n"
        f"Parser error: {error}\n\n"
        "Invalid JSON:\n"
        f"{bad_text}"
    )
