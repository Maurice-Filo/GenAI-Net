"""Opt-in full-duplex Harness adapter; frozen no-LLM streams remain unchanged."""

from RL4CRN.llm.harness_runner import HarnessDeciderWriterCRNGraph
from RL4CRN.llm.schemas import CandidateEvaluation
from RL4CRN.llm.workspace_loss_protocol import PROTOCOL
from RL4CRN.search_exploration.engine import atomic_json
from RL4CRN.search_exploration.harness import CachedHarnessEvaluator, CachedEvaluation, ExplorationHarnessClient
from RL4CRN.search_exploration.evaluation import materialize
from RL4CRN.search_exploration.store import BudgetExhausted
from .julia_stream import JuliaStreamTrainer


class StreamHarnessEvaluator(CachedHarnessEvaluator):
    """Separate scientific rejection from an infrastructure failure."""

    fatal_error = None

    def evaluate(self, candidate):
        prepared = self.base._prepare_candidate(candidate)
        if isinstance(prepared, CandidateEvaluation):
            return prepared
        ids, parameters = prepared
        values = {"reaction_ids": list(ids), "rates": [float(p[0]) for p in parameters]}
        try:
            record, hit = self.service.evaluate(values, category="llm")
            env = materialize(self.base, values)
            env.state.last_task_info = {**record["task_info"], "reward": record["loss"], "source": "LLM"}
            complete = bool(record["simulation_complete"])
            return CachedEvaluation(candidate=candidate, valid=complete, env=env, loss=record["loss"],
                message=("valid; cached" if hit else "valid") if complete else "canonical_incomplete",
                task_info=env.state.last_task_info, raw_actions=env.raw_actions_taken,
                exploration_record={**values, **record})
        except BudgetExhausted as error:
            return CandidateEvaluation(candidate=candidate, valid=False, message=f"BudgetExhausted: {error}")
        except Exception as error:
            self.fatal_error = error
            raise RuntimeError("Harness canonical evaluator infrastructure failed") from error


class HarnessJuliaStreamTrainer(JuliaStreamTrainer):
    method_name = "search-streams-julia-harness-v1"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.llm_fit_candidates = []
        self.llm_rounds_merged = 0
        self.llm_candidates_merged = 0
        self._withholding_context = False

    def attach_harness(self, *, dsh_home, prompt, every=5, candidates=10,
                       model="deepseek-v4-flash", request_timeout=900.):
        self.harness_evaluator = StreamHarnessEvaluator(self.evaluator, self.service)
        client = ExplorationHarnessClient(
            workspace_root=self.output / "harness-workspaces", dsh_home=dsh_home,
            model=model, provider="deepseek-official", timeout_seconds=request_timeout,
            candidate_validation_policy="independent-members", harness_protocol=PROTOCOL,
        )
        graph = HarnessDeciderWriterCRNGraph(
            client=client, evaluator=self.harness_evaluator, writer_retry_limit=0,
            max_workspace_evaluations=0, max_literature_searches=0,
        )
        self.configure_llm_graph(
            graph, every=every, task_description=prompt, num_candidates=candidates,
            withhold_initial_hof=True, cross_communication=True,
            add_to_hall_of_fame=True, max_in_flight=1,
            jsonl_path=self.output / "llm_evaluations.jsonl",
        )

    def _sil_feedback_summary(self):
        if self._withholding_context:
            return "Request 0: RL/HOF/EIL information intentionally withheld."
        return super()._sil_feedback_summary()

    def forbidden_topology_summary(self, limit=10):
        if self._withholding_context:
            return "Request 0: no search-derived exclusions supplied; static reaction masks still apply."
        return super().forbidden_topology_summary(limit)

    def _maybe_run_llm_graph(self, epoch):
        self._check_harness_error()
        self._withholding_context = bool(self._llm_loop and epoch == self._llm_loop["start_epoch"])
        try:
            return super()._maybe_run_llm_graph(epoch)
        finally:
            self._withholding_context = False

    def _check_harness_error(self):
        evaluator = getattr(self, "harness_evaluator", None)
        if evaluator is not None and evaluator.fatal_error is not None:
            raise RuntimeError("Harness canonical evaluator infrastructure failed") from evaluator.fatal_error

    def _record(self, record, source, proposal_id):
        result = super()._record(record, source, proposal_id)
        # The legacy snapshot helper reads 'source', not exploration_source.
        for env in self.s.mult_env.hall_of_fame:
            info = env.state.last_task_info
            if info.get("exploration_source"):
                info["source"] = info["exploration_source"].upper()
        if source == "llm" and record.get("stream") != "optimization":
            self.llm_fit_candidates.append({**record, "source": "llm", "proposal_id": proposal_id,
                                            "stream": "llm", "parameter_operator": None})
            self.llm_candidates_merged += 1
        return result

    def _collect_llm_graph_result(self, **kwargs):
        result = super()._collect_llm_graph_result(**kwargs)
        self._check_harness_error()
        if result is not None:
            self.llm_rounds_merged += 1
            for index, evaluation in enumerate(result.evaluations):
                record = getattr(evaluation, "exploration_record", None)
                if record is not None and not evaluation.valid and evaluation.message == "canonical_incomplete":
                    # A bad initial trajectory is not proof of a bad topology.
                    self.llm_fit_candidates.append({**record, "source": "llm", "origin": "llm",
                        "parent_hash": None, "policy_version": None, "stream": "llm",
                        "parameter_operator": None,
                        "proposal_id": f"llm-incomplete-{self.state.epoch}-{index}-{record['candidate_hash'][:10]}"})
        if self._llm_loop:
            atomic_json(self.output / "llm_history.json", self._llm_loop["history"])
        return result

    def _schedule(self, records):
        # Compete under the same admission limits, not an extra fitting budget.
        candidates, self.llm_fit_candidates = self.llm_fit_candidates, []
        return super()._schedule([*records, *candidates])

    def metrics(self, *args, **kwargs):
        result = super().metrics(*args, **kwargs)
        settings = self._llm_loop or {}
        result.update({"LLM/pending_rounds": len(settings.get("jobs", [])),
                       "LLM/merged_rounds": self.llm_rounds_merged,
                       "LLM/merged_candidates": self.llm_candidates_merged,
                       "Source/llm/cumulative_gain": self.external_gain})
        return result

    def finish(self):
        self.persist_hof_boundary(epoch=self.state.epoch, boundary="training_boundary",
            rl_evaluations=0, llm_final_evaluations=0, llm_tool_evaluations=0, total_evaluations=0)
        atomic_json(self.output / "status.json", {"status": "draining_llm", "epoch": self.state.epoch})
        self.wait_for_llm_graph(timeout=None)
        self._check_harness_error()
        # Record late arrivals honestly: they cannot influence completed RL updates.
        self._merge_ready(drain=True)
        self._schedule([])
        super().finish()

    def persist_hof_boundary(self, **kwargs):
        # The frozen no-LLM parent passes zero LLM counters at its terminal boundary.
        counts = self._snapshot_evaluation_counts(self.state.epoch)
        kwargs.update(rl_evaluations=counts["rl"], llm_final_evaluations=counts["llm_final"],
                      llm_tool_evaluations=counts["llm_tool"], total_evaluations=counts["total"])
        return super().persist_hof_boundary(**kwargs)

    def close_streams(self):
        self.clear_llm_graph(wait=True)
        super().close_streams()
