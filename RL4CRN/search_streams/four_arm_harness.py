"""Four communication arms; frozen two-arm and full-duplex runners are unchanged."""

import json

from RL4CRN.search_exploration.engine import atomic_json
from RL4CRN.utils.input_interface import Trainer
from .directional_harness import DirectionalHarnessTrainer
from .harness import HarnessJuliaStreamTrainer


MODES = ("full", "llm-to-rl", "rl-to-llm", "none")


def channels(mode):
    if mode not in MODES:
        raise ValueError("Unknown communication arm")
    return mode in ("full", "llm-to-rl"), mode in ("full", "rl-to-llm")


class FourArmHarnessTrainer(DirectionalHarnessTrainer):
    method_name = "search-streams-julia-four-arm-v1"

    def attach_harness(self, *, communication_mode, **kwargs):
        inward, outward = channels(communication_mode)
        self.communication_mode = communication_mode
        HarnessJuliaStreamTrainer.attach_harness(self, **kwargs)
        self._llm_loop.update(add_to_hall_of_fame=inward, cross_communication=outward)

    def _record(self, record, source, proposal_id):
        if not channels(self.communication_mode)[0] and (
            source == "llm" or record.get("origin") == "llm"
        ):
            raise RuntimeError("Uncoupled arm attempted to admit an LLM lineage to search")
        return HarnessJuliaStreamTrainer._record(self, record, source, proposal_id)

    def _schedule(self, records):
        if not channels(self.communication_mode)[0] and (
            self.llm_fit_candidates or any(
                r.get("origin") == "llm" or r.get("source") == "llm" for r in records
            )
        ):
            raise RuntimeError("Uncoupled arm attempted to fit an LLM proposal")
        return HarnessJuliaStreamTrainer._schedule(self, records)

    def _collect_llm_graph_result(self, **kwargs):
        settings = self._llm_loop
        if not settings:
            return None
        if channels(self.communication_mode)[0]:
            result = HarnessJuliaStreamTrainer._collect_llm_graph_result(self, **kwargs)
        else:
            # Keep self-memory/provenance while bypassing archive admission and fitting.
            result = Trainer._collect_llm_graph_result(self, **kwargs)
            self._check_harness_error()
            if result is not None:
                self.llm_rounds_merged += 1
            settings["isolated_candidates"].clear()
        atomic_json(self.output / "llm_history.json", settings["history"])
        if result is None:
            return None
        history = settings["history"][-1]
        slots = (result.response_validation or {}).get("accepted_candidate_indices", [])
        with (self.output / "direct_llm_proposals.jsonl").open("a") as stream:
            for index, evaluation in enumerate(result.evaluations):
                record = getattr(evaluation, "exploration_record", None)
                scalar = {k: v for k, v in record.items() if k != "task_info"} if record else None
                if scalar is not None:
                    self.direct_llm_records.append(scalar)
                row = {"launched_epoch": history["launched_epoch"],
                    "completed_epoch": history["completed_epoch"],
                    "candidate_index": slots[index] if index < len(slots) else index,
                    "valid": bool(evaluation.valid), "message": evaluation.message,
                    "candidate": evaluation.candidate.to_dict(), "record": scalar,
                    "direct_supplied_rate_only": True, "communication_mode": self.communication_mode}
                stream.write(json.dumps(row, sort_keys=True) + "\n")
        return result
