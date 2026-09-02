"""Optional LLM-assisted CRN generation utilities.

The classes exported here do not change the RL training path.  They provide an
extra way to propose candidate CRNs and then score them through the same
``Environment``/``LibraryActuator``/``IOCRNStepper`` interfaces used by RL.
"""

from RL4CRN.llm.candidate_evaluator import LLMCandidateEvaluator
from RL4CRN.llm.benchmark_prompts import (
    CRN_AGENT_SYSTEM_PROMPT,
    MMC2_LOGIC_TASK_PROMPT,
    MMC2_LOGIC_TRAJECTORY_TASK_PROMPT,
    MMC2_RPA_TASK_PROMPT,
    MMC2_TASK_PROMPTS,
    get_mmc2_task_prompt,
    get_mmc2_task_prompt_variant,
    get_reported_mmc2_task_prompt_2026,
)
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
from RL4CRN.llm.harness_client import (
    HarnessLLMClient,
    HarnessResponseError,
    HarnessRunWorkspace,
    HarnessUnavailableError,
    build_crn_output_contract,
)
from RL4CRN.llm.harness_runner import HarnessCRNGenerator, HarnessDeciderWriterCRNGraph
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
    "CRN_AGENT_SYSTEM_PROMPT",
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
    "MMC2_LOGIC_TASK_PROMPT",
    "MMC2_LOGIC_TRAJECTORY_TASK_PROMPT",
    "MMC2_RPA_TASK_PROMPT",
    "MMC2_TASK_PROMPTS",
    "GeminiAPIKeyLLMClient",
    "HarnessCRNGenerator",
    "HarnessDeciderWriterCRNGraph",
    "HarnessLLMClient",
    "HarnessResponseError",
    "HarnessRunWorkspace",
    "HarnessUnavailableError",
    "VertexAIUnavailableError",
    "VertexCRNGenerator",
    "VertexLLMClient",
    "candidates_to_payload",
    "build_crn_output_contract",
    "default_decider_writer_spec",
    "get_mmc2_task_prompt",
    "get_mmc2_task_prompt_variant",
    "get_reported_mmc2_task_prompt_2026",
    "parse_candidates_payload",
    "plot_llm_evaluations",
]
