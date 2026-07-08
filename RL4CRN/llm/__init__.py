"""Optional LLM-assisted CRN generation utilities.

The classes exported here do not change the RL training path.  They provide an
extra way to propose candidate CRNs and then score them through the same
``Environment``/``LibraryActuator``/``IOCRNStepper`` interfaces used by RL.
"""

from RL4CRN.llm.candidate_evaluator import LLMCandidateEvaluator
from RL4CRN.llm.debate import DebateTranscript, LLMCRNDebate
from RL4CRN.llm.generator import LLMCRNGenerator, LLMGenerationRound, VertexCRNGenerator
from RL4CRN.llm.graphs import (
    DeciderWriterCRNGraph,
    LLMGraphNode,
    LLMGraphRunResult,
    LLMGraphSpec,
    default_decider_writer_spec,
    plot_llm_evaluations,
)
from RL4CRN.llm.memory import LLMFeedbackRecord, LLMMemory
from RL4CRN.llm.schemas import (
    CandidateEvaluation,
    LLMCandidate,
    LLMGenerationConfig,
    candidates_to_payload,
    parse_candidates_payload,
)
from RL4CRN.llm.vertex_client import (
    DEFAULT_GEMINI_MODEL,
    GeminiAPIKeyLLMClient,
    VertexAIUnavailableError,
    VertexLLMClient,
)

__all__ = [
    "CandidateEvaluation",
    "DEFAULT_GEMINI_MODEL",
    "DebateTranscript",
    "DeciderWriterCRNGraph",
    "LLMCandidate",
    "LLMCandidateEvaluator",
    "LLMCRNDebate",
    "LLMCRNGenerator",
    "LLMFeedbackRecord",
    "LLMGraphNode",
    "LLMGraphRunResult",
    "LLMGraphSpec",
    "LLMGenerationConfig",
    "LLMGenerationRound",
    "LLMMemory",
    "GeminiAPIKeyLLMClient",
    "VertexAIUnavailableError",
    "VertexCRNGenerator",
    "VertexLLMClient",
    "candidates_to_payload",
    "default_decider_writer_spec",
    "parse_candidates_payload",
    "plot_llm_evaluations",
]
