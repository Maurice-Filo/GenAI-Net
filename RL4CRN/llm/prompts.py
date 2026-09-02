"""Prompt builders for LLM-assisted CRN generation.

These helpers are intentionally provider-neutral.  They turn RL4CRN objects into
plain text and define the JSON contract expected from the language model.
"""

from __future__ import annotations

from typing import Any, Iterable, Optional

from RL4CRN.llm.benchmark_prompts import CRN_AGENT_SYSTEM_PROMPT


DEFAULT_SYSTEM_PROMPT = CRN_AGENT_SYSTEM_PROMPT


def format_reaction_library(reaction_library: Any) -> str:
    """Format a reaction library as an ID-indexed menu."""

    reactions = getattr(reaction_library, "reactions", [])
    if not reactions:
        return "The reaction library is empty."
    return "\n".join(f"ID {idx}: {reaction}" for idx, reaction in enumerate(reactions))


def format_hall_of_fame(hall_of_fame_iter: Optional[Iterable[Any]], limit: int = 5) -> str:
    """Format current RL Hall-of-Fame entries for a prompt."""

    if hall_of_fame_iter is None:
        return "No Hall-of-Fame entries are available yet."

    entries = list(hall_of_fame_iter)[: int(limit)]
    if not entries:
        return "The Hall of Fame is empty."

    lines = []
    for i, env in enumerate(entries, start=1):
        info = getattr(env.state, "last_task_info", {}) or {}
        loss = info.get("reward", "N/A")
        lines.append(f"--- Hall-of-Fame #{i}; loss={loss} ---\n{env.state}")
    return "\n".join(lines)


def build_candidate_generation_prompt(
    *,
    task_description: str,
    reaction_library: Any,
    max_added_reactions: int,
    num_candidates: int,
    hall_of_fame_iter: Optional[Iterable[Any]] = None,
    feedback_text: str = "",
    llm_best_text: str = "",
    forbidden_topologies_text: str = "",
    sil_feedback_text: str = "",
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    hall_of_fame_limit: int = 5,
) -> str:
    """Build the JSON-generation prompt used by the single-agent generator."""

    hof_text = format_hall_of_fame(hall_of_fame_iter, limit=hall_of_fame_limit)
    library_text = format_reaction_library(reaction_library)
    feedback_text = feedback_text or "No recent LLM feedback is available."
    llm_best_text = llm_best_text or "No prior LLM-generated best candidates are available."
    forbidden_topologies_text = forbidden_topologies_text or "No forbidden topologies have been archived yet."
    sil_feedback_text = sil_feedback_text or "No completed SIL update is available yet."
    novel_count = max(1, (int(num_candidates) * 3 + 4) // 5)
    refinement_count = int(num_candidates) - novel_count
    search_mix_text = (
        f"Produce {novel_count} candidates with new reaction-ID sets and "
        f"{refinement_count} candidates that refine parameters of promising "
        "Hall-of-Fame reaction-ID sets. A Hall-of-Fame set remains eligible for "
        "refinement unless it is explicitly present in the forbidden archive."
        if refinement_count > 0
        else "Produce one new candidate informed by the Hall of Fame and evaluator feedback."
    )

    return f"""{system_prompt}

=== Task ===
{task_description}

=== Reaction Budget ===
Use exactly {max_added_reactions} added reactions for each candidate.
Do not repeat a reaction ID within the same candidate.

=== Recent LLM Feedback ===
{feedback_text}

=== Best LLM-Generated Candidates ===
{llm_best_text}

=== RL Hall of Fame ===
{hof_text}

=== RL Self-Imitation Learning Status ===
This is optimization context, not a candidate-quality score.
{sil_feedback_text}

=== Exploration And Refinement Mix ===
{search_mix_text}
Never return two candidates with identical reaction IDs and parameter values.

=== Forbidden Already-Evaluated Topologies ===
The following topologies were already evaluated, archived, and are no longer
admissible for this search. Do not propose candidates with the same reaction-ID
set, even with different parameters.
{forbidden_topologies_text}

=== Available Reaction Library ===
Select reactions only by the IDs listed here.
{library_text}

=== Output Contract ===
Generate {num_candidates} new, distinct candidates. Return only one JSON
object, with no markdown fences, no comments, and no text before or after it:
{{
  "candidates": [
    {{
      "reaction_ids": [0, 3, 5],
      "parameter_values": [[1.0], [0.5, 2.0], [0.1]]
    }}
  ]
}}

Each entry in "parameter_values" must be the full parameter vector for the
corresponding reaction ID. Use positive finite numerical parameters. The number
of reaction IDs must exactly match the number of parameter vectors."""
