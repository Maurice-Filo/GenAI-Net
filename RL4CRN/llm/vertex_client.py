"""Optional VertexAI/Gemini client for LLM-assisted CRN generation."""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

from RL4CRN.llm.schemas import LLMGenerationConfig


class VertexAIUnavailableError(ImportError):
    """Raised when VertexAI support is requested but not installed."""


class VertexLLMClient:
    """Thin wrapper around VertexAI Gemini models.

    The import of ``vertexai`` is intentionally lazy.  This keeps RL4CRN usable
    on machines without Google Cloud dependencies or billing access; the client
    only fails when a Vertex model is actually constructed.
    """

    def __init__(
        self,
        *,
        project_id: str,
        location: str = "europe-west1",
        model_name: str = "gemini-2.5-flash",
    ):
        self.project_id = project_id
        self.location = location
        self.model_name = model_name

        try:
            import vertexai
            from vertexai.generative_models import GenerativeModel
        except ImportError as exc:
            raise VertexAIUnavailableError(
                "VertexAI support requires the optional Google Cloud SDK. "
                "Install RL4CRN with the 'vertexai' extra or install "
                "'google-cloud-aiplatform' in this environment."
            ) from exc

        vertexai.init(project=project_id, location=location)
        self._model = GenerativeModel(model_name)

    def generate_json(
        self,
        prompt: str,
        *,
        generation_config: Optional[LLMGenerationConfig] = None,
    ) -> Dict[str, Any]:
        """Generate JSON from a prompt and parse the model response."""

        try:
            from vertexai.generative_models import GenerationConfig
        except ImportError as exc:
            raise VertexAIUnavailableError("VertexAI generation classes are unavailable.") from exc

        cfg = generation_config or LLMGenerationConfig()
        response = self._model.generate_content(
            prompt,
            generation_config=GenerationConfig(**cfg.to_dict()),
        )
        text = getattr(response, "text", "")
        return json.loads(text)
