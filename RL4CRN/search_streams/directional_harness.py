"""Directional interventions isolated from the frozen full-duplex implementation."""

import json
import math

from RL4CRN.search_exploration.engine import atomic_json
from RL4CRN.utils.input_interface import Trainer
from .harness import HarnessJuliaStreamTrainer


MODES = ("llm-to-rl", "rl-to-llm")


def topology_union(search, proposals, threshold):
    """Pool complete, evaluated records by structural hash, never by rate vector."""
    pools = []
    for records in (search, proposals):
        best = {}
        for record in records:
            if not record.get("simulation_complete"):
                continue
            if not math.isfinite(record["loss"]) or record["loss"] < 0:
                raise ValueError("Invalid loss in a completed union record")
            key = record["topology_hash"]
            if key not in best or record["loss"] < best[key]["loss"]:
                best[key] = record
        pools.append(best)
    merged = dict(pools[0])
    for key, record in pools[1].items():
        if key not in merged or record["loss"] < merged[key]["loss"]:
            merged[key] = record
    qualified = [{k for k, r in pool.items() if r["loss"] < threshold} for pool in (*pools, merged)]
    return {"search_qualified": len(qualified[0]), "llm_qualified": len(qualified[1]),
        "overlap_qualified": len(qualified[0] & qualified[1]),
        "union_qualified": len(qualified[2]), "additional_qualified": len(qualified[2] - qualified[0]),
        "search_best_loss": min((r["loss"] for r in pools[0].values()), default=None),
        "llm_best_loss": min((r["loss"] for r in pools[1].values()), default=None),
        "union_best_loss": min((r["loss"] for r in merged.values()), default=None)}


class DirectionalHarnessTrainer(HarnessJuliaStreamTrainer):
    method_name = "search-streams-julia-directional-v1"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.communication_mode = None
        self.direct_llm_records = []
        self.terminal_union = None

    def attach_harness(self, *, communication_mode, **kwargs):
        if communication_mode not in MODES:
            raise ValueError("Expected a directional communication arm")
        self.communication_mode = communication_mode
        super().attach_harness(**kwargs)
        self._llm_loop.update(add_to_hall_of_fame=communication_mode == "llm-to-rl",
                              cross_communication=communication_mode == "rl-to-llm")

    def _record(self, record, source, proposal_id):
        if self.communication_mode == "rl-to-llm" and (source == "llm" or record.get("origin") == "llm"):
            raise RuntimeError("Return-only arm attempted to admit an LLM lineage to search")
        return super()._record(record, source, proposal_id)

    def _schedule(self, records):
        if self.communication_mode == "rl-to-llm":
            if self.llm_fit_candidates or any(r.get("origin") == "llm" or r.get("source") == "llm" for r in records):
                raise RuntimeError("Return-only arm attempted to fit an LLM proposal")
        return super()._schedule(records)

    def _collect_llm_graph_result(self, **kwargs):
        settings = self._llm_loop
        if not settings:
            return None
        if self.communication_mode == "llm-to-rl":
            result = super()._collect_llm_graph_result(**kwargs)
        else:
            # Bypass both archive insertion and the inherited incomplete-proposal fitting queue.
            result = Trainer._collect_llm_graph_result(self, **kwargs)
            self._check_harness_error()
            if result is not None:
                self.llm_rounds_merged += 1
            # The scalar ledger retains independent outputs without keeping trajectory clones.
            settings["isolated_candidates"].clear()
            atomic_json(self.output / "llm_history.json", settings["history"])
        if result is None:
            return None
        history = settings["history"][-1]
        validation = result.response_validation or {}
        slots = validation.get("accepted_candidate_indices", [])
        for evaluation in result.evaluations:
            record = getattr(evaluation, "exploration_record", None)
            if record is not None:
                self.direct_llm_records.append({k: v for k, v in record.items() if k != "task_info"})
        with (self.output / "direct_llm_proposals.jsonl").open("a") as stream:
            for index, evaluation in enumerate(result.evaluations):
                record = getattr(evaluation, "exploration_record", None)
                row = {"launched_epoch": history["launched_epoch"],
                    "completed_epoch": history["completed_epoch"],
                    "candidate_index": slots[index] if index < len(slots) else index,
                    "valid": bool(evaluation.valid), "message": evaluation.message,
                    "candidate": evaluation.candidate.to_dict(),
                    "record": {k: v for k, v in record.items() if k != "task_info"} if record is not None else None,
                    "direct_supplied_rate_only": True, "communication_mode": self.communication_mode}
                stream.write(json.dumps(row, sort_keys=True) + "\n")
        return result

    def metrics(self, *args, **kwargs):
        metrics = super().metrics(*args, **kwargs)
        metrics["LLM/received_rounds"] = self.llm_rounds_merged
        metrics["LLM/direct_evaluated_records"] = len(self.direct_llm_records)
        if self.terminal_union is not None:
            metrics.update({"Union/qualified_topologies": self.terminal_union["union_qualified"],
                            "Union/best_loss": self.terminal_union["union_best_loss"]})
        return metrics

    def finish(self):
        super().finish()
        search = (json.loads(row[0]) for row in self.store.db.execute("SELECT payload FROM designs"))
        self.terminal_union = topology_union(search, self.direct_llm_records, self.config.quality_threshold)
        atomic_json(self.output / "terminal_union.json", self.terminal_union)
        self.last_metrics = self.metrics()
        self.telemetry.metrics(self.state.epoch, self.last_metrics)
        # Union reporting does not mutate the training HOF, archive, visits or replay state.
