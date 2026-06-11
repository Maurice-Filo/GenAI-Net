"""Prompt builders for LLM-assisted CRN generation.

These helpers are intentionally provider-neutral.  They turn RL4CRN objects into
plain text and define the JSON contract expected from the language model.
"""

from __future__ import annotations

from typing import Any, Iterable, Optional


DEFAULT_SYSTEM_PROMPT = """You are an expert in chemical reaction networks, synthetic biology, and control theory.
You propose CRNs by selecting reactions from a fixed library and assigning numerical parameters.
Return strictly valid JSON and respect the exact reaction budget."""


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
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    hall_of_fame_limit: int = 5,
) -> str:
    """Build the JSON-generation prompt used by the single-agent generator."""

    hof_text = format_hall_of_fame(hall_of_fame_iter, limit=hall_of_fame_limit)
    library_text = format_reaction_library(reaction_library)
    feedback_text = feedback_text or "No recent LLM feedback is available."
    llm_best_text = llm_best_text or "No prior LLM-generated best candidates are available."

    return f"""{system_prompt}

=== Task ===
{task_description}

=== Reaction Budget ===
Use exactly {max_added_reactions} added reactions for each candidate.

=== Recent LLM Feedback ===
{feedback_text}

=== Best LLM-Generated Candidates ===
{llm_best_text}

=== RL Hall of Fame ===
{hof_text}

=== Available Reaction Library ===
Select reactions only by the IDs listed here.
{library_text}

=== Output Contract ===
Generate {num_candidates} new, distinct candidates.  Return one JSON object:
{{
  "candidates": [
    {{
      "reasoning": "brief mechanistic rationale",
      "reaction_ids": [0, 3, 5],
      "parameter_values": [[1.0], [0.5, 2.0], [0.1]]
    }}
  ]
}}

Each entry in "parameter_values" must be the full parameter vector for the
corresponding reaction ID.  Use positive finite numerical parameters."""
