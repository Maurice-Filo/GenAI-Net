import json
import sqlite3
import tempfile
import threading
import time
import unittest
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from RL4CRN.agent2env_interface.iocrn_stepper import IOCRNStepper
from RL4CRN.agent2env_interface.library_actuator import LibraryActuator
from RL4CRN.llm import (
    DeciderWriterCRNGraph,
    LLMCandidate,
    LLMCandidateEvaluator,
    LLMGenerationConfig,
    LLMGraphNode,
    LLMGraphSpec,
    LLMCRNGenerator,
    parse_candidates_payload,
)
from RL4CRN.utils.forbidden_topologies import ForbiddenTopologyArchive
from RL4CRN.utils.hall_of_fame import HallOfFame
from RL4CRN.utils.input_interface import Trainer
from RL4CRN.utils.parameter_optimization import ParameterOptimizationResult
from RL4CRN.utils.results_database import ResultsDatabase, serialize_crn
from RL4CRN.utils.results_viewer import ResultsDatabaseReader


class ToyReaction:
    def __init__(self, reaction_id, num_parameters=1):
        self.ID = reaction_id
        self.num_parameters = num_parameters
        self.params = [None] * num_parameters

    def set_parameters(self, params):
        if len(params) != self.num_parameters:
            raise AssertionError("wrong number of parameters")
        self.params = list(params)

    def __str__(self):
        return f"R{self.ID}(k_{self.ID})"


class ToyLibrary:
    def __init__(self):
        self.reactions = [ToyReaction(0, 1), ToyReaction(1, 2), ToyReaction(2, 1)]

    def __len__(self):
        return len(self.reactions)

    def get_reaction(self, idx):
        rxn = self.reactions[idx]
        return ToyReaction(rxn.ID, rxn.num_parameters)


class ToyCRN:
    num_unknown_params = 0
    num_inputs = 0
    num_reactions = 0

    def __init__(self):
        self.reactions = []
        self.last_task_info = {}

    def clone(self):
        out = ToyCRN()
        out.reactions = list(self.reactions)
        out.last_task_info = dict(self.last_task_info)
        return out

    def add_reaction(self, reaction):
        self.reactions.append(reaction)

    def compile(self):
        return None

    def get_bool_signature(self):
        ids = [reaction.ID for reaction in self.reactions]
        return np.array(ids, dtype=np.int64)

    def gather_reaction_IDs(self):
        return [reaction.ID for reaction in self.reactions]

    def __str__(self):
        return ", ".join(str(reaction) for reaction in self.reactions)


def toy_reward(state):
    total = 0.0
    for reaction in state.reactions:
        total += sum(reaction.params)
    return total, {"num_added": len(state.reactions), "u": np.array([0.0, 1.0])}


def build_evaluator(**kwargs):
    library = ToyLibrary()
    initial_reaction_ids = kwargs.pop("initial_reaction_ids", ())
    crn_template = ToyCRN()
    for reaction_id in initial_reaction_ids:
        crn_template.add_reaction(library.get_reaction(reaction_id))
    return LLMCandidateEvaluator(
        crn_template=crn_template,
        max_added_reactions=2,
        library=library,
        stepper=IOCRNStepper(),
        actuator=LibraryActuator(library),
        compute_reward_func=toy_reward,
        **kwargs,
    )


class FakeClient:
    def __init__(self, payload):
        self.payload = payload
        self.prompts = []

    def generate_json(self, prompt, generation_config=None):
        self.prompts.append(prompt)
        return self.payload

    def generate_text(self, prompt, generation_config=None):
        self.prompts.append(prompt)
        return "strategy: use the toy reactions without duplicates"


class SequencedFakeClient(FakeClient):
    def __init__(self, payloads):
        super().__init__(payload=None)
        self.payloads = list(payloads)

    def generate_json(self, prompt, generation_config=None):
        self.prompts.append(prompt)
        payload = self.payloads.pop(0)
        if isinstance(payload, Exception):
            raise payload
        return payload


class BlockingGraph:
    def __init__(self, result):
        self.result = result
        self.started = threading.Event()
        self.release = threading.Event()
        self.kwargs = None

    def run_round(self, **kwargs):
        self.kwargs = kwargs
        self.started.set()
        if not self.release.wait(timeout=5.0):
            raise TimeoutError("test did not release background LLM graph")
        return self.result


class ForkableBlockingGraph:
    def __init__(self, result, shared=None):
        self.result = result
        self.shared = shared or SimpleNamespace(
            lock=threading.Lock(),
            started=[],
            release=threading.Event(),
        )

    def fork(self):
        return type(self)(self.result, self.shared)

    def run_round(self, **kwargs):
        with self.shared.lock:
            self.shared.started.append(kwargs["step"])
        if not self.shared.release.wait(timeout=5.0):
            raise TimeoutError("test did not release concurrent LLM graphs")
        return self.result


class FakeLogger:
    def __init__(self):
        self.metrics = []
        self.texts = []
        self.assets = []

    def log_metric(self, name, value, step=None):
        self.metrics.append((name, value, step))

    def log_text(self, text):
        self.texts.append(text)

    def log_asset_data(self, data, name=None, step=None):
        self.assets.append((name, data, step))


@dataclass
class TrainCfg:
    max_added_reactions: int = 2


@dataclass
class PolicyCfg:
    ordering_enabled: bool = False


@dataclass
class Cfg:
    train: TrainCfg
    policy: PolicyCfg


class SessionLike:
    def __init__(self):
        self.crn_template = ToyCRN()
        self.library = ToyLibrary()
        self.stepper = IOCRNStepper()
        self.actuator = LibraryActuator(self.library)
        self.task = type("Task", (), {"compute_reward": staticmethod(toy_reward)})()
        self.cfg = Cfg(train=TrainCfg(), policy=PolicyCfg())
        self.logger = None


class LLMIntegrationTests(unittest.TestCase):
    def test_results_database_saves_hof_snapshot_without_llm(self):
        evaluation = build_evaluator().evaluate(
            LLMCandidate([0, 1], [[1.0], [2.0, 3.0]], "direct test candidate")
        )
        self.assertTrue(evaluation.valid, evaluation.message)
        hof = HallOfFame(max_size=3)
        evaluation.env.state.last_task_info["outputs"] = [np.array([[0.1, 0.2, 0.4]])]
        hof.add(evaluation.env)
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        database_path = f"{temp_dir.name}/results.sqlite"
        database = ResultsDatabase(database_path, run_id="hof-only-test")

        database.record_hof_snapshot(hof, epoch=7, save_plots=True)
        database.close()

        with sqlite3.connect(database_path) as connection:
            snapshot = connection.execute(
                "SELECT run_id, epoch FROM hof_snapshots"
            ).fetchone()
            entry = connection.execute(
                "SELECT rank, loss, topology_hash FROM hof_snapshot_entries"
            ).fetchone()
            self.assertEqual(snapshot, ("hof-only-test", 7))
            self.assertEqual(entry[0], 0)
            self.assertEqual(entry[1], 6.0)
            self.assertTrue(entry[2])
        plot = Path(temp_dir.name) / "hof-plots" / f"{entry[2]}.jpg"
        self.assertTrue(plot.is_file())
        self.assertEqual(plot.read_bytes()[:2], b"\xff\xd8")
        reader = ResultsDatabaseReader(database_path)
        self.assertEqual(reader.hof_plot(entry[2]), plot.resolve())

    def test_results_database_preserves_invalid_llm_candidate(self):
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        database_path = f"{temp_dir.name}/results.sqlite"
        database = ResultsDatabase(database_path, run_id="invalid-llm-test")
        result = SimpleNamespace(
            response_validation={
                "returned_candidate_count": 2,
                "accepted_candidate_count": 1,
                "accepted_candidate_indices": [0],
                "rejected_candidates": [{"candidate_index": 1, "error": "duplicate"}],
                "clamped_parameter_count": 2,
                "provider_call_count": 2,
            },
            evaluations=[
                SimpleNamespace(
                    candidate=LLMCandidate([0, 0], [[1.0], [2.0]], "duplicate"),
                    valid=False,
                    loss=None,
                    env=None,
                    message="candidate contains duplicate reaction IDs.",
                    task_info={},
                )
            ]
        )

        database.record_llm_round(result, launched_epoch=3, requested=1)
        database.close()

        with sqlite3.connect(database_path) as connection:
            row = connection.execute(
                "SELECT valid, topology_hash, message FROM llm_candidates"
            ).fetchone()
            self.assertEqual(row[0], 0)
            self.assertIsNone(row[1])
            self.assertIn("duplicate", row[2])
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM evaluations").fetchone()[0], 0)
        llm_run = ResultsDatabaseReader(database_path).llm_runs("invalid-llm-test")[0]
        self.assertEqual(llm_run["returned"], 2)
        self.assertEqual(llm_run["accepted"], 1)
        self.assertEqual(llm_run["rejected"], 1)
        self.assertEqual(llm_run["clamped_parameters"], 2)
        self.assertEqual(llm_run["provider_call_count"], 2)

    def test_results_database_preserves_failed_llm_round_validation(self):
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        database_path = f"{temp_dir.name}/results.sqlite"
        database = ResultsDatabase(database_path, run_id="failed-llm-test")

        database.record_llm_failure(
            launched_epoch=20,
            completed_epoch=31,
            requested=10,
            elapsed_seconds=4.5,
            error="all Writer members were invalid",
            response_validation={
                "returned_candidate_count": 3,
                "accepted_candidate_count": 0,
                "rejected_candidates": [
                    {"candidate_index": 0, "reason": "fixed template reaction"},
                    {"candidate_index": 1, "reason": "duplicate"},
                    {"candidate_index": 2, "reason": "non-finite rate"},
                ],
                "clamped_parameter_count": 2,
            },
        )
        database.close()

        reader = ResultsDatabaseReader(database_path)
        row = reader.llm_runs("failed-llm-test")[0]
        self.assertEqual(row["status"], "failed")
        self.assertEqual(row["produced"], 3)
        self.assertEqual(row["valid_count"], 0)
        self.assertEqual(row["rejected"], 3)
        self.assertEqual(row["clamped_parameters"], 2)
        self.assertIn("invalid", row["error"])
        self.assertEqual(reader.summary("failed-llm-test")["llm_failure_count"], 1)

    def test_hof_provenance_distinguishes_direct_exact_and_parameter_refinement(self):
        evaluator = build_evaluator()
        direct = evaluator.evaluate(
            LLMCandidate([0, 1], [[1.0], [2.0, 3.0]], "direct")
        )
        direct.env.state.last_task_info.update(
            {
                "source": "LLM",
                "emitter": "LLM",
                "llm_proposal_id": "epoch-0:writer-member-0",
                "llm_first_seen_epoch": 0,
            }
        )
        identity = serialize_crn(direct.env.state)
        llm_provenance = {
            identity["topology_hash"]: {
                "topology_first_emitter": "LLM",
                "first_proposal_id": "epoch-0:writer-member-0",
                "first_seen_epoch": 0,
                "candidate_hashes": {identity["candidate_hash"]},
                "exposed_to_rl": True,
            }
        }
        exact = direct.env.clone()
        exact.state.last_task_info["source"] = "RL"
        exact.state.last_task_info["emitter"] = "RL"
        refined = evaluator.evaluate(
            LLMCandidate([0, 1], [[4.0], [5.0, 6.0]], "refined")
        )
        refined.env.state.last_task_info.update({"source": "RL", "emitter": "RL"})

        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        database_path = f"{temp_dir.name}/results.sqlite"
        database = ResultsDatabase(database_path, run_id="provenance-test")
        database.record_hof_snapshot([direct.env], epoch=0, llm_provenance=llm_provenance)
        database.record_hof_snapshot([exact], epoch=1, llm_provenance=llm_provenance)
        database.record_hof_snapshot([refined.env], epoch=2, llm_provenance=llm_provenance)
        database.close()

        with sqlite3.connect(database_path) as connection:
            rows = connection.execute(
                """SELECT h.epoch, e.emitter, e.provenance_class,
                          e.related_llm_proposal_id
                     FROM hof_snapshot_entries e
                     JOIN hof_snapshots h ON h.snapshot_id = e.snapshot_id
                     ORDER BY h.epoch"""
            ).fetchall()
        self.assertEqual(
            rows,
            [
                (0, "LLM", "direct_llm", "epoch-0:writer-member-0"),
                (1, "RL", "rl_exact_reemission_of_llm_candidate", "epoch-0:writer-member-0"),
                (2, "RL", "rl_parameter_refinement_of_llm_topology", "epoch-0:writer-member-0"),
            ],
        )

    def test_parse_candidates_payload(self):
        candidates = parse_candidates_payload(
            {
                "candidates": [
                    {
                        "reasoning": "test",
                        "reaction_ids": [0, 1],
                        "parameter_values": [[1.0], [2.0, 3.0]],
                    }
                ]
            }
        )
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].reaction_ids, [0, 1])
        self.assertEqual(candidates[0].parameter_values, [[1.0], [2.0, 3.0]])

    def test_evaluate_valid_candidate_through_rl_interfaces(self):
        evaluator = build_evaluator()
        result = evaluator.evaluate(
            LLMCandidate(
                reaction_ids=[0, 1],
                parameter_values=[[1.5], [2.0, 3.0]],
                reasoning="valid",
            )
        )
        self.assertTrue(result.valid, result.message)
        self.assertEqual(result.loss, 6.5)
        self.assertEqual(result.task_info["num_added"], 2)
        self.assertEqual(result.task_info["source"], "LLM")
        self.assertEqual([action["reaction index"] for action in result.raw_actions], [0, 1])

    def test_invalid_candidates_are_rejected_before_simulation(self):
        evaluator = build_evaluator()
        bad_id = evaluator.evaluate(LLMCandidate([99, 0], [[1.0], [1.0]], "bad id"))
        bad_count = evaluator.evaluate(LLMCandidate([0], [[1.0]], "short"))
        bad_params = evaluator.evaluate(LLMCandidate([1, 0], [[1.0], [1.0]], "bad params"))
        bad_duplicate = evaluator.evaluate(LLMCandidate([0, 0], [[1.0], [1.0]], "duplicate"))

        self.assertFalse(bad_id.valid)
        self.assertIn("outside the library", bad_id.message)
        self.assertFalse(bad_count.valid)
        self.assertIn("expected 2", bad_count.message)
        self.assertFalse(bad_params.valid)
        self.assertIn("expects 2 parameters", bad_params.message)
        self.assertFalse(bad_duplicate.valid)
        self.assertIn("duplicate reaction IDs", bad_duplicate.message)

    def test_template_reaction_ids_are_forbidden_to_llm_candidates(self):
        evaluator = build_evaluator(initial_reaction_ids=[1])

        result = evaluator.evaluate(
            LLMCandidate([1, 0], [[2.0, 3.0], [1.0]], "duplicates template ID")
        )

        self.assertFalse(result.valid)
        self.assertIn("reaction ID 1 is forbidden", result.message)
        self.assertEqual(evaluator.initial_reaction_ids, frozenset({1}))
        self.assertIn(1, evaluator.forbidden_reaction_ids)

    def test_ordered_policy_sorts_candidate_before_rollout(self):
        evaluator = build_evaluator(is_ordered_policy=True)
        result = evaluator.evaluate(
            LLMCandidate(
                reaction_ids=[1, 0],
                parameter_values=[[2.0, 3.0], [1.0]],
                reasoning="sort me",
            )
        )
        self.assertTrue(result.valid, result.message)
        self.assertEqual([action["reaction index"] for action in result.raw_actions], [0, 1])

    def test_evaluator_rejects_forbidden_topology(self):
        archive = ForbiddenTopologyArchive()
        evaluator = build_evaluator(forbidden_topologies=archive, forbidden_loss=999.0)
        candidate = LLMCandidate(
            reaction_ids=[0, 1],
            parameter_values=[[1.0], [2.0, 3.0]],
            reasoning="archive me",
        )
        first = evaluator.evaluate(candidate)
        self.assertTrue(first.valid, first.message)
        archive.add_state(first.env.state, loss=first.loss, epoch=0, rank=0)

        second = evaluator.evaluate(candidate)
        self.assertFalse(second.valid)
        self.assertEqual(second.loss, 999.0)
        self.assertIn("forbidden topology", second.message)

    def test_generator_uses_fake_client_and_can_log_and_update_hof(self):
        payload = {
            "candidates": [
                {
                    "reasoning": "generated",
                    "reaction_ids": [0, 1],
                    "parameter_values": [[1.0], [2.0, 3.0]],
                }
            ]
        }
        session = SessionLike()
        generator = LLMCRNGenerator.from_session(client=FakeClient(payload), session=session)
        hof = HallOfFame(max_size=3)

        with tempfile.NamedTemporaryFile(suffix=".jsonl") as tmp:
            round_result = generator.run_round(
                task_description="Minimize toy loss.",
                hall_of_fame_iter=[],
                add_to_hall_of_fame=hof,
                jsonl_path=tmp.name,
            )
            with open(tmp.name, encoding="utf-8") as handle:
                records = [json.loads(line) for line in handle]

        self.assertEqual(len(round_result.evaluations), 1)
        self.assertTrue(round_result.evaluations[0].valid)
        self.assertEqual(len(hof), 1)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["task_info"]["u"], [0.0, 1.0])
        self.assertIn("Minimize toy loss.", round_result.prompt)

    def test_decider_writer_graph_runs_and_logs_llm_metrics(self):
        payload = {
            "candidates": [
                {
                    "reasoning": "generated by graph",
                    "reaction_ids": [0, 1],
                    "parameter_values": [[1.0], [2.0, 3.0]],
                }
            ]
        }
        session = SessionLike()
        evaluator = LLMCandidateEvaluator.from_session(session)
        logger = FakeLogger()
        spec = LLMGraphSpec(
            nodes=[
                LLMGraphNode(
                    "Decider",
                    "text",
                    "Task={task_description}; budget={max_added_reactions}; feedback={feedback_text}; best={llm_best_text}",
                    LLMGenerationConfig(response_mime_type="text/plain"),
                ),
                LLMGraphNode("Writer", "json", "Task={task_description}; decision={decision}"),
            ],
            edges=[("Decider", "Writer")],
            decider_node="Decider",
            writer_node="Writer",
        )
        graph = DeciderWriterCRNGraph(
            client=FakeClient(payload),
            evaluator=evaluator,
            spec=spec,
            comet_logger=logger,
        )

        hof = HallOfFame(max_size=3)
        with tempfile.NamedTemporaryFile(suffix=".jsonl") as tmp:
            result = graph.run_round(
                task_description="Minimize toy loss.",
                num_candidates=1,
                add_to_hall_of_fame=hof,
                jsonl_path=tmp.name,
                step=2,
            )
            with open(tmp.name, encoding="utf-8") as handle:
                records = [json.loads(line) for line in handle]

        self.assertEqual(len(result.evaluations), 1)
        self.assertTrue(result.evaluations[0].valid)
        self.assertEqual(result.evaluations[0].env.state.last_task_info["source"], "LLM")
        self.assertEqual(len(hof), 1)
        self.assertEqual(hof[0].state.last_task_info["source"], "LLM")
        self.assertEqual(records[0]["task_info"]["source"], "LLM")
        metric_names = {name for name, _, _ in logger.metrics}
        self.assertIn("LLM/Loss Best", metric_names)
        self.assertIn("LLM/Loss Candidate 0", metric_names)
        self.assertIn("LLM/Timing Round Seconds", metric_names)
        self.assertIn("LLM/Timing Decider Seconds", metric_names)
        self.assertIn("LLM/Timing Writer Generate Seconds", metric_names)
        self.assertIn("LLM/Timing Candidate Evaluation Seconds", metric_names)
        self.assertTrue(any("Decider -> Writer" in text for text in logger.texts))
        self.assertTrue(any(name.startswith("llm_candidates_step_") for name, _, _ in logger.assets))
        self.assertTrue(any(name.startswith("llm_discussion_step_") for name, _, _ in logger.assets))
        self.assertTrue(any(name.startswith("llm_payload_and_evaluation_step_") for name, _, _ in logger.assets))

    def test_decider_writer_graph_retries_after_writer_error(self):
        good_payload = {
            "candidates": [
                {
                    "reaction_ids": [0, 1],
                    "parameter_values": [[1.0], [2.0, 3.0]],
                }
            ]
        }
        session = SessionLike()
        evaluator = LLMCandidateEvaluator.from_session(session)
        logger = FakeLogger()
        client = SequencedFakeClient([ValueError("invalid JSON"), good_payload])
        graph = DeciderWriterCRNGraph(client=client, evaluator=evaluator, comet_logger=logger)

        hof = HallOfFame(max_size=3)
        result = graph.run_round(
            task_description="Minimize toy loss.",
            num_candidates=1,
            add_to_hall_of_fame=hof,
            step=5,
        )

        self.assertEqual(len(client.prompts), 3)  # decider, failed writer, retry writer
        self.assertIn("Previous Attempt Failed", client.prompts[-1])
        self.assertTrue(result.evaluations[0].valid)
        self.assertEqual(len(hof), 1)
        self.assertTrue(any("Writer retry feedback" in text for text in logger.texts))

    def test_decider_writer_graph_saves_local_message_transcript(self):
        payload = {
            "candidates": [
                {
                    "reaction_ids": [0, 1],
                    "parameter_values": [[1.0], [2.0, 3.0]],
                }
            ]
        }
        session = SessionLike()
        evaluator = LLMCandidateEvaluator.from_session(session)
        logger = FakeLogger()

        with tempfile.TemporaryDirectory() as tmpdir:
            transcript_path = f"{tmpdir}/messages.jsonl"
            graph = DeciderWriterCRNGraph(
                client=FakeClient(payload),
                evaluator=evaluator,
                comet_logger=logger,
                transcript_jsonl_path=transcript_path,
            )
            graph.run_round(task_description="Minimize toy loss.", num_candidates=1, step=0)

            with open(transcript_path, encoding="utf-8") as handle:
                records = [json.loads(line) for line in handle]

        edges = {record["edge"] for record in records}
        kinds = {record["kind"] for record in records}
        self.assertIn(
            "Task contract + Search constraints + RL Hall of Fame + RL SIL status + Feedback memory + Forbidden topology archive -> Decider",
            edges,
        )
        self.assertIn("Decider -> Writer", edges)
        self.assertIn("Writer -> Evaluator", edges)
        self.assertIn("Evaluator -> Feedback memory", edges)
        self.assertIn("prompt", kinds)
        self.assertIn("json_payload", kinds)
        self.assertTrue(any(name.endswith(".json") for name, _, _ in logger.assets))

    def test_decider_writer_graph_does_not_insert_forbidden_llm_candidate(self):
        payload = {
            "candidates": [
                {
                    "reaction_ids": [0, 1],
                    "parameter_values": [[1.0], [2.0, 3.0]],
                }
            ]
        }
        archive = ForbiddenTopologyArchive()
        session = SessionLike()
        evaluator = LLMCandidateEvaluator.from_session(session)
        first = evaluator.evaluate(LLMCandidate([0, 1], [[1.0], [2.0, 3.0]], "seed archive"))
        archive.add_state(first.env.state, loss=first.loss, epoch=0, rank=0)
        evaluator.forbidden_topologies = archive

        graph = DeciderWriterCRNGraph(
            client=SequencedFakeClient([payload, payload]),
            evaluator=evaluator,
        )
        hof = HallOfFame(max_size=3)
        result = graph.run_round(
            task_description="Minimize toy loss.",
            num_candidates=1,
            add_to_hall_of_fame=hof,
            step=0,
        )

        self.assertEqual(len(hof), 0)
        self.assertFalse(result.evaluations[0].valid)
        self.assertTrue(result.evaluations[0].task_info["forbidden_topology"])

    def test_trainer_llm_hook_uses_requested_candidate_count_and_hof(self):
        payload = {
            "candidates": [
                {
                    "reasoning": "generated by trainer hook",
                    "reaction_ids": [0, 1],
                    "parameter_values": [[1.0], [2.0, 3.0]],
                }
            ]
        }
        session_like = SessionLike()
        evaluator = LLMCandidateEvaluator.from_session(session_like)
        logger = FakeLogger()
        graph = DeciderWriterCRNGraph(
            client=FakeClient(payload),
            evaluator=evaluator,
            comet_logger=logger,
        )

        trainer = Trainer(
            SimpleNamespace(
                mult_env=SimpleNamespace(hall_of_fame=HallOfFame(max_size=3)),
                logger=logger,
            )
        )
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        database_path = f"{temp_dir.name}/results.sqlite"
        trainer.configure_results_database(
            database_path,
            every=2,
            run_id="trainer-llm-test",
            metadata={"task": "toy-rpa"},
        )
        trainer.configure_llm_graph(
            graph,
            every=2,
            task_description="Minimize toy loss.",
            num_candidates=10,
        )

        self.assertIsNone(trainer._maybe_run_llm_graph(1))
        launched = trainer._maybe_run_llm_graph(2)
        result = trainer.wait_for_llm_graph(timeout=2.0)

        self.assertIsNone(launched)
        self.assertIsNotNone(result)
        trainer._maybe_persist_hof(2)
        trainer.close_results_database()
        self.assertEqual(len(trainer.s.mult_env.hall_of_fame), 1)
        self.assertEqual(trainer.s.mult_env.hall_of_fame[0].state.last_task_info["source"], "LLM")
        self.assertEqual(trainer.llm_graph_history()[0]["requested"], 10)
        metric_names = {name for name, _, _ in logger.metrics}
        self.assertIn("LLM/Requested Count", metric_names)
        self.assertIn("LLM/Hall of Fame Size After", metric_names)
        self.assertIn("LLM/Timing Trainer Hook Seconds", metric_names)
        with sqlite3.connect(database_path) as connection:
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM llm_runs").fetchone()[0], 1)
            self.assertEqual(
                connection.execute("SELECT COUNT(*) FROM llm_candidates").fetchone()[0], 1
            )
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM evaluations").fetchone()[0], 1)
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM hof_snapshots").fetchone()[0], 1)
        reader = ResultsDatabaseReader(database_path)
        self.assertEqual(reader.runs()[0]["task"], "toy-rpa")
        self.assertEqual(reader.summary("trainer-llm-test")["topology_count"], 1)
        history = reader.loss_history("trainer-llm-test")
        self.assertEqual(history[0]["dominant_source"], "LLM")
        llm_runs = reader.llm_runs("trainer-llm-test")
        self.assertEqual(llm_runs[0]["best_loss"], 6.0)
        llm_candidates = reader.llm_candidates(llm_runs[0]["llm_run_id"])
        self.assertEqual(len(llm_candidates), 1)
        self.assertIn("presentation", llm_candidates[0])
        self.assertIn("reactions", llm_candidates[0]["presentation"])
        hof_entries = reader.latest_hof("trainer-llm-test")["entries"]
        saved_crns = reader.crns("trainer-llm-test")
        self.assertEqual(len(hof_entries), 1)
        self.assertEqual(hof_entries[0]["initial_source"], "LLM")
        self.assertEqual(len(saved_crns), 1)
        self.assertEqual(saved_crns[0]["initial_source"], "LLM")
        self.assertEqual(
            reader.crn_detail(saved_crns[0]["topology_hash"])["initial_source"],
            "LLM",
        )
        trainer.clear_llm_graph(wait=True)

    def test_trainer_llm_hook_does_not_block_rl_and_merges_on_main_thread(self):
        evaluator = build_evaluator()
        evaluation = evaluator.evaluate(
            LLMCandidate([0, 1], [[1.0], [2.0, 3.0]], "background candidate")
        )
        result = SimpleNamespace(evaluations=[evaluation])
        graph = BlockingGraph(result)
        trainer = Trainer(
            SimpleNamespace(
                mult_env=SimpleNamespace(hall_of_fame=HallOfFame(max_size=3)),
                logger=FakeLogger(),
            )
        )
        trainer.configure_llm_graph(
            graph,
            every=1,
            task_description="Minimize toy loss.",
            num_candidates=1,
        )

        started = time.perf_counter()
        self.assertIsNone(trainer._maybe_run_llm_graph(0))
        hook_seconds = time.perf_counter() - started

        self.assertTrue(graph.started.wait(timeout=1.0))
        self.assertLess(hook_seconds, 0.2)
        self.assertEqual(len(trainer.s.mult_env.hall_of_fame), 0)
        rl_work_completed = sum(range(1000))
        self.assertEqual(rl_work_completed, 499500)

        graph.release.set()
        merged = trainer.wait_for_llm_graph(timeout=2.0)

        self.assertIs(merged, result)
        self.assertEqual(len(trainer.s.mult_env.hall_of_fame), 1)
        self.assertEqual(trainer.llm_graph_history()[0]["launched_epoch"], 0)
        trainer.clear_llm_graph(wait=True)

    def test_no_communication_retains_llm_candidate_until_terminal_pooling(self):
        evaluator = build_evaluator()
        rl_evaluation = evaluator.evaluate(
            LLMCandidate([0, 1], [[1.0], [2.0, 3.0]], "RL-side candidate")
        )
        llm_evaluation = evaluator.evaluate(
            LLMCandidate([0, 2], [[1.0], [1.0]], "isolated LLM candidate")
        )
        hof = HallOfFame(max_size=3)
        hof.add(rl_evaluation.env)
        graph = BlockingGraph(SimpleNamespace(evaluations=[llm_evaluation]))
        trainer = Trainer(
            SimpleNamespace(mult_env=SimpleNamespace(hall_of_fame=hof), logger=FakeLogger())
        )
        trainer.configure_llm_graph(
            graph,
            every=1,
            task_description="Minimize toy loss.",
            num_candidates=1,
            add_to_hall_of_fame=False,
            cross_communication=False,
        )

        trainer._maybe_run_llm_graph(0)
        self.assertTrue(graph.started.wait(timeout=1.0))
        self.assertEqual(tuple(graph.kwargs["hall_of_fame_iter"]), ())
        self.assertIn("withheld", graph.kwargs["sil_feedback_text"])
        self.assertIn("withheld", graph.kwargs["forbidden_topologies_text"])
        graph.release.set()
        trainer.wait_for_llm_graph(timeout=2.0)

        self.assertEqual(len(hof), 1)
        self.assertEqual(trainer.merge_isolated_llm_candidates(), 1)
        self.assertEqual(len(hof), 2)
        self.assertEqual(trainer.merge_isolated_llm_candidates(), 0)
        trainer.clear_llm_graph(wait=True)

    def test_initial_hof_withholding_preserves_later_cross_communication(self):
        evaluator = build_evaluator()
        evaluation = evaluator.evaluate(
            LLMCandidate([0, 1], [[1.0], [2.0, 3.0]], "candidate")
        )
        hof = HallOfFame(max_size=3)
        hof.add(evaluation.env)
        graph = BlockingGraph(SimpleNamespace(evaluations=[evaluation]))
        trainer = Trainer(
            SimpleNamespace(mult_env=SimpleNamespace(hall_of_fame=hof), logger=FakeLogger())
        )
        trainer.configure_llm_graph(
            graph,
            every=2,
            task_description="Minimize toy loss.",
            num_candidates=1,
            withhold_initial_hof=True,
        )

        trainer._maybe_run_llm_graph(0)
        self.assertTrue(graph.started.wait(timeout=1.0))
        self.assertEqual(tuple(graph.kwargs["hall_of_fame_iter"]), ())
        graph.release.set()
        trainer.wait_for_llm_graph(timeout=2.0)

        graph.started.clear()
        trainer._maybe_run_llm_graph(2)
        self.assertTrue(graph.started.wait(timeout=1.0))
        self.assertGreater(len(tuple(graph.kwargs["hall_of_fame_iter"])), 0)
        trainer.wait_for_llm_graph(timeout=2.0)
        trainer.clear_llm_graph(wait=True)

    def test_trainer_launches_each_cadence_with_an_earlier_request_pending(self):
        evaluator = build_evaluator()
        evaluation = evaluator.evaluate(
            LLMCandidate([0, 1], [[1.0], [2.0, 3.0]], "concurrent candidate")
        )
        result = SimpleNamespace(evaluations=[evaluation], tool_evaluations=[])
        graph = ForkableBlockingGraph(result)
        trainer = Trainer(
            SimpleNamespace(
                mult_env=SimpleNamespace(hall_of_fame=HallOfFame(max_size=3)),
                logger=FakeLogger(),
            )
        )
        trainer.configure_llm_graph(
            graph,
            every=2,
            task_description="Minimize toy loss.",
            num_candidates=1,
            max_in_flight=2,
        )

        self.assertIsNone(trainer._maybe_run_llm_graph(0))
        self.assertIsNone(trainer._maybe_run_llm_graph(2))
        deadline = time.monotonic() + 1.0
        while len(graph.shared.started) < 2 and time.monotonic() < deadline:
            time.sleep(0.01)

        self.assertEqual(sorted(graph.shared.started), [0, 2])
        self.assertEqual(len(trainer._llm_loop["jobs"]), 2)
        graph.shared.release.set()
        self.assertIs(trainer.wait_for_llm_graph(timeout=2.0), result)
        self.assertEqual(
            [record["launched_epoch"] for record in trainer.llm_graph_history()],
            [0, 2],
        )
        self.assertEqual(len(trainer._llm_loop["jobs"]), 0)
        trainer.clear_llm_graph(wait=True)

    def test_writer_prompt_contains_hof_sil_and_ten_candidate_search_mix(self):
        evaluator = build_evaluator()
        evaluation = evaluator.evaluate(
            LLMCandidate([0, 1], [[1.0], [2.0, 3.0]], "hall candidate")
        )
        hof = HallOfFame(max_size=3)
        hof.add(evaluation.env)
        graph = DeciderWriterCRNGraph(client=FakeClient({"candidates": []}), evaluator=evaluator)

        prompt = graph.build_writer_prompt(
            task_description="Minimize toy loss.",
            decision="Explore and refine.",
            num_candidates=10,
            hall_of_fame_iter=hof,
            sil_feedback_text="enabled=True; step=4; hall_of_fame_size=1; sil_loss=0.25",
        )

        self.assertIn("Hall-of-Fame #1", prompt)
        self.assertIn("sil_loss=0.25", prompt)
        self.assertIn("Produce 6 candidates with new reaction-ID sets", prompt)
        self.assertIn("4 candidates that refine parameters", prompt)

    def test_forbidden_archive_uses_threshold_when_ipopt_unavailable(self):
        session_like = SessionLike()
        evaluator = LLMCandidateEvaluator.from_session(session_like)
        evaluation = evaluator.evaluate(
            LLMCandidate([0, 1], [[1.0], [2.0, 3.0]], "candidate")
        )
        self.assertTrue(evaluation.valid, evaluation.message)

        hof = HallOfFame(max_size=3)
        hof.add(evaluation.env)
        train_cfg = SimpleNamespace(
            forbidden_topology_m=1,
            forbidden_topology_every=1,
            forbidden_topology_start_epoch=0,
            forbidden_threshold=10.0,
            forbidden_optimize_with_ipopt=True,
            forbidden_ipopt_maxiter=2,
            forbidden_ipopt_log_min=-18.0,
            forbidden_ipopt_log_max=6.0,
        )
        trainer = Trainer(
            SimpleNamespace(
                mult_env=SimpleNamespace(hall_of_fame=hof),
                logger=FakeLogger(),
                cfg=SimpleNamespace(train=train_cfg),
                task=SimpleNamespace(compute_reward=toy_reward),
            )
        )
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        database_path = f"{temp_dir.name}/results.sqlite"
        trainer.configure_results_database(
            database_path,
            every=5,
            run_id="trainer-optimization-test",
        )

        added = trainer._refresh_forbidden_topologies(epoch=0)
        self.assertEqual(added, 1)
        self.assertEqual(len(trainer.s.forbidden_topologies), 1)
        metric_names = {name for name, _, _ in trainer.s.logger.metrics}
        self.assertIn("Forbidden Topologies/Timing Total Seconds", metric_names)
        self.assertIn("Forbidden Topologies/Optimization Seconds", metric_names)
        self.assertIn("Forbidden Topologies/Optimization Evaluations", metric_names)
        archive_assets = [
            json.loads(data)
            for name, data, _ in trainer.s.logger.assets
            if name and name.startswith("forbidden_topology_archive_epoch_")
        ]
        self.assertTrue(archive_assets)
        self.assertIn("optimization_seconds", archive_assets[-1])
        self.assertIn("optimization_evaluations", archive_assets[-1])

        trainer.s.forbidden_topologies.records.clear()
        train_cfg.forbidden_threshold = 1.0
        added = trainer._refresh_forbidden_topologies(epoch=1)
        self.assertEqual(added, 0)
        self.assertEqual(len(trainer.s.forbidden_topologies), 0)
        trainer.close_results_database()
        with sqlite3.connect(database_path) as connection:
            self.assertEqual(
                connection.execute("SELECT COUNT(*) FROM optimization_runs").fetchone()[0], 2
            )
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM evaluations").fetchone()[0], 2)
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM crns").fetchone()[0], 1)

    def test_enforced_exclusion_optimizes_in_background_and_merges_at_boundary(self):
        evaluation = build_evaluator().evaluate(
            LLMCandidate([0, 1], [[1.0], [2.0, 3.0]], "candidate")
        )
        hof = HallOfFame(max_size=3)
        hof.add(evaluation.env)
        cfg = SimpleNamespace(
            forbidden_topology_m=1,
            forbidden_topology_every=1,
            forbidden_topology_start_epoch=0,
            forbidden_async=True,
            forbidden_ipopt_maxiter=10,
            forbidden_ipopt_log_min=-18.0,
            forbidden_ipopt_log_max=6.0,
            forbidden_optimization_max_evaluations=50,
            forbidden_optimization_timeout_seconds=120.0,
        )
        trainer = Trainer(
            SimpleNamespace(
                mult_env=SimpleNamespace(hall_of_fame=hof),
                logger=FakeLogger(),
                cfg=SimpleNamespace(train=cfg),
                task=SimpleNamespace(compute_reward=toy_reward),
            )
        )
        release = threading.Event()

        def bounded_result(state, *_args, **_kwargs):
            release.wait(2)
            return ParameterOptimizationResult(True, True, 0.2, state, "done", 7)

        with patch(
            "RL4CRN.utils.input_interface.optimize_crn_parameters_ipopt",
            side_effect=bounded_result,
        ):
            self.assertEqual(trainer._refresh_forbidden_topologies(0), 0)
            self.assertEqual(len(trainer.s.forbidden_topologies), 0)
            release.set()
            self.assertEqual(trainer.wait_for_forbidden_topologies(), 1)

        self.assertEqual(len(trainer.s.forbidden_topologies), 1)
        self.assertEqual(trainer.forbidden_optimization_evaluations(), 7)


if __name__ == "__main__":
    unittest.main()
