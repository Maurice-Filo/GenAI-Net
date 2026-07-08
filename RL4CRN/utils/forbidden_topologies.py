"""Forbidden-topology archive for dynamic CRN search exclusion."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Optional, Tuple


def topology_signature_key(state: Any) -> bytes:
    """Return the canonical exact-topology key used by the Hall of Fame.

    The key is the byte representation of ``state.get_bool_signature()``.  This
    ignores parameter values and reaction ordering, but preserves labelled
    species and the selected reaction set.
    """

    signature = state.get_bool_signature()
    return signature.tobytes()


@dataclass
class ForbiddenTopologyRecord:
    """One archived topology excluded from future search."""

    signature: bytes
    loss: float
    epoch: int
    rank: int
    source: str = "hall_of_fame"
    crn: str = ""
    optimization_attempted: bool = False
    optimization_success: bool = False
    optimization_message: str = ""

    def to_prompt_text(self) -> str:
        return (
            f"- forbidden topology rank={self.rank}, archived_epoch={self.epoch}, "
            f"loss={self.loss:.6g}, source={self.source}\n{self.crn}"
        )


@dataclass
class ForbiddenTopologyArchive:
    """Store exact CRN topologies that should no longer be admissible."""

    records: Dict[bytes, ForbiddenTopologyRecord] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.records)

    def contains_state(self, state: Any) -> bool:
        return topology_signature_key(state) in self.records

    def add_state(
        self,
        state: Any,
        *,
        loss: float,
        epoch: int,
        rank: int,
        source: str = "hall_of_fame",
        optimization_attempted: bool = False,
        optimization_success: bool = False,
        optimization_message: str = "",
    ) -> bool:
        """Add a topology, returning True only when it was newly archived."""

        key = topology_signature_key(state)
        if key in self.records:
            return False
        self.records[key] = ForbiddenTopologyRecord(
            signature=key,
            loss=float(loss),
            epoch=int(epoch),
            rank=int(rank),
            source=str(source),
            crn=str(state),
            optimization_attempted=bool(optimization_attempted),
            optimization_success=bool(optimization_success),
            optimization_message=str(optimization_message),
        )
        return True

    def add_from_hall_of_fame(self, hall_of_fame: Optional[Iterable[Any]], *, m: int, epoch: int) -> int:
        """Archive the top ``m`` currently ranked Hall-of-Fame topologies."""

        if hall_of_fame is None or m <= 0:
            return 0
        added = 0
        for rank, env in enumerate(hall_of_fame):
            if rank >= int(m):
                break
            info = getattr(env.state, "last_task_info", {}) or {}
            loss = float(info.get("reward", float("nan")))
            if self.add_state(env.state, loss=loss, epoch=epoch, rank=rank):
                added += 1
        return added

    def signature_set(self) -> frozenset[bytes]:
        """Return a multiprocessing-friendly snapshot for reward wrappers."""

        return frozenset(self.records.keys())

    def format_for_prompt(self, limit: int = 10) -> str:
        """Return a compact LLM-facing list of excluded topologies."""

        if not self.records:
            return "No forbidden topologies have been archived yet."
        ordered = sorted(self.records.values(), key=lambda r: (r.epoch, r.rank))
        shown = ordered[-int(limit) :]
        return "\n".join(record.to_prompt_text() for record in shown)


def reward_with_forbidden_topologies(
    state: Any,
    reward_fn: Any,
    forbidden_signatures: frozenset[bytes],
    forbidden_loss: float,
) -> Tuple[float, Dict[str, Any]]:
    """Return a penalty for forbidden topologies, otherwise delegate reward."""

    if forbidden_signatures and topology_signature_key(state) in forbidden_signatures:
        return float(forbidden_loss), {
            "reward": float(forbidden_loss),
            "forbidden_topology": True,
            "forbidden_reason": "topology already archived as evaluated/admissible solution",
        }

    result = reward_fn(state)
    if isinstance(result, tuple):
        return result
    return result, {}
