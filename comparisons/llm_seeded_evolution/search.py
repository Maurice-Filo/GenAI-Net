"""Small evolutionary optimizer, independent of the RL trainer and LLM clients."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Iterable

import numpy as np


@dataclass(frozen=True)
class Candidate:
    reaction_ids: tuple[int, ...]
    rates: tuple[float, ...]
    source: str = "random"
    parent_hash: str | None = None
    proposal_id: str | None = None


@dataclass(frozen=True)
class ScoredCandidate:
    candidate: Candidate
    loss: float
    topology_hash: str
    crn_text: str = ""


class EliteArchive:
    """Keep the best parameterization of each of the best labelled topologies."""

    def __init__(self, capacity: int):
        if capacity < 1:
            raise ValueError("Archive capacity must be positive.")
        self.capacity = capacity
        self.entries: dict[str, ScoredCandidate] = {}

    def ranked(self) -> list[ScoredCandidate]:
        return sorted(self.entries.values(), key=lambda x: (x.loss, x.topology_hash))

    def add(self, item: ScoredCandidate) -> None:
        if not math.isfinite(item.loss):
            raise ValueError("Canonical evaluator must return a finite loss, including its failure loss.")
        previous = self.entries.get(item.topology_hash)
        if previous is not None and previous.loss <= item.loss:
            return
        self.entries[item.topology_hash] = item
        if len(self.entries) > self.capacity:
            del self.entries[self.ranked()[-1].topology_hash]


class Evolution:
    def __init__(
        self, *, seed: int, allowed_ids: Iterable[int], n_reactions: int,
        archive_size: int, rate_range: tuple[float, float] = (0.1, 50.0),
        rate_probability: float = 0.45, topology_probability: float = 0.45,
        log_rate_sigma: float = 0.5,
    ):
        self.rng = np.random.default_rng(seed)
        self.allowed_ids = tuple(sorted(set(allowed_ids)))
        if not 1 <= n_reactions < len(self.allowed_ids):
            raise ValueError("Need more allowed reactions than the positive reaction budget.")
        if not (0 <= rate_probability <= 1 and 0 <= topology_probability <= 1
                and rate_probability + topology_probability <= 1):
            raise ValueError("Mutation probabilities must be nonnegative and sum to at most one.")
        if not (0 < rate_range[0] <= rate_range[1] < math.inf and 0 < log_rate_sigma < math.inf):
            raise ValueError("Rate prior and mutation scale must be finite and positive.")
        self.n_reactions = n_reactions
        self.rate_range = rate_range
        self.rate_probability = rate_probability
        self.topology_probability = topology_probability
        self.log_rate_sigma = log_rate_sigma
        self.archive = EliteArchive(archive_size)

    def fresh_rates(self, n: int) -> tuple[float, ...]:
        return tuple(float(x) for x in np.exp(self.rng.uniform(
            np.log(self.rate_range[0]), np.log(self.rate_range[1]), n)))

    def propose(self) -> Candidate:
        draw = self.rng.random()
        parents = self.archive.ranked()
        if not parents or draw >= self.rate_probability + self.topology_probability:
            return Candidate(tuple(int(x) for x in self.rng.choice(
                self.allowed_ids, self.n_reactions, replace=False)), self.fresh_rates(self.n_reactions))
        parent = parents[int(self.rng.integers(len(parents)))]
        ids, rates = list(parent.candidate.reaction_ids), list(parent.candidate.rates)
        if draw < self.rate_probability:
            # One rate per offspring; no artificial [0.001,100] bound on search.
            i = int(self.rng.integers(len(rates)))
            value = float(np.exp(np.log(rates[i]) + self.rng.normal(0, self.log_rate_sigma)))
            if not 0 < value < math.inf:
                raise FloatingPointError("Rate mutation exceeded floating-point support.")
            rates[i] = value
            source = "rate_mutation"
        else:
            i = int(self.rng.integers(len(ids)))
            available = [rid for rid in self.allowed_ids if rid not in ids]
            ids[i] = int(self.rng.choice(available))
            rates[i] = self.fresh_rates(1)[0]
            source = "reaction_mutation"
        return Candidate(tuple(ids), tuple(rates), source, parent.topology_hash,
                         parent.candidate.proposal_id)


def run_search(
    evolution: Evolution, *, events: list[dict], search_budget: int, batch_size: int,
    evaluate_batch: Callable[[list[Candidate]], list[ScoredCandidate]],
    record: Callable[[ScoredCandidate, int, int], None],
    checkpoint: Callable[[int, int], None] | None = None,
) -> tuple[int, int]:
    """Replay at search-evaluation boundaries, never exposing future proposals.

    Batched offspring use the archive at the start of their batch. Batches are
    split at injection boundaries; results are consumed in proposal order, so
    changing worker count does not change the stochastic search trajectory.
    """
    if search_budget < 1 or batch_size < 1:
        raise ValueError("Search budget and evolutionary batch size must be positive.")
    if any(int(e["after_search_evaluations"]) < 0 for e in events):
        raise ValueError("Negative proposal availability boundary.")
    pending = sorted(events, key=lambda e: (e["after_search_evaluations"], e["launched_epoch"]))
    search_count = llm_count = event_index = 0

    def consume(candidates: list[Candidate], *, llm: bool) -> None:
        nonlocal search_count, llm_count
        results = evaluate_batch(candidates)
        if len(results) != len(candidates):
            raise RuntimeError("Evaluator returned an incomplete batch.")
        for expected, result in zip(candidates, results):
            if result.candidate != expected:
                raise RuntimeError("Evaluator changed candidate order or parameters.")
            search_count += int(not llm)
            llm_count += int(llm)
            evolution.archive.add(result)
            record(result, search_count, llm_count)

    while True:
        while event_index < len(pending) and pending[event_index]["after_search_evaluations"] <= search_count:
            event = pending[event_index]
            candidates = [Candidate(tuple(c["reaction_ids"]), tuple(c["rates"]), "llm",
                                    proposal_id=c["proposal_id"]) for c in event["candidates"]]
            consume(candidates, llm=True)
            event_index += 1
        if checkpoint is not None:
            checkpoint(search_count, llm_count)
        if search_count == search_budget:
            return search_count, llm_count
        next_event = (pending[event_index]["after_search_evaluations"]
                      if event_index < len(pending) else search_budget)
        count = min(batch_size, search_budget - search_count, next_event - search_count)
        consume([evolution.propose() for _ in range(count)], llm=False)
