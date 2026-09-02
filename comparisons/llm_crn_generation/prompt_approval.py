"""Fail-closed author approval gate for contract-v2 model prompts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROMPT_REVIEW = (
    ROOT
    / "paper/iclr2027_genai_net_llm/generated/CONTRACT_V2_PROMPT_REVIEW.json"
)
DEFAULT_PROMPT_APPROVAL = (
    ROOT
    / "paper/iclr2027_genai_net_llm/generated/CONTRACT_V2_PROMPT_APPROVAL.json"
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_prompt_approval(
    approval_path: str | Path = DEFAULT_PROMPT_APPROVAL,
    *,
    review_path: str | Path = DEFAULT_PROMPT_REVIEW,
) -> Dict[str, Any]:
    """Return approval metadata only when it matches the current review packet."""

    approval_path = Path(approval_path).expanduser().resolve()
    review_path = Path(review_path).expanduser().resolve()
    if not review_path.is_file():
        raise RuntimeError(f"Prompt review packet is missing: {review_path}")
    if not approval_path.is_file():
        raise RuntimeError(
            "Paper campaign blocked pending author prompt approval. Review "
            f"{review_path.with_suffix('.md')} and create {approval_path} from the "
            "generated non-approving example only after accepting every prompt."
        )
    try:
        approval = json.loads(approval_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Invalid prompt approval file: {approval_path}") from exc
    expected_review_hash = file_sha256(review_path)
    if approval.get("approval_status") != "approved":
        raise RuntimeError("Prompt approval status must be exactly 'approved'.")
    if approval.get("prompt_review_sha256") != expected_review_hash:
        raise RuntimeError(
            "Prompt approval does not match the current review packet; review the changed "
            "prompts and issue a new approval."
        )
    for field in ("approved_by", "approved_at"):
        if not str(approval.get(field, "")).strip():
            raise RuntimeError(f"Prompt approval field {field!r} must be non-empty.")
    return {
        **approval,
        "approval_file": str(approval_path),
        "approval_file_sha256": file_sha256(approval_path),
        "prompt_review_file": str(review_path),
    }
