import json
import tempfile
import unittest
from dataclasses import dataclass
from types import SimpleNamespace

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

    def __str__(self):
        return ", ".join(str(reaction) for reaction in self.reactions)


def toy_reward(state):
    total = 0.0
    for reaction in state.reactions:
        total += sum(reaction.params)
    return total, {"num_added": len(state.reactions), "u": np.array([0.0, 1.0])}


def build_evaluator(**kwargs):
    library = ToyLibrary()
    return LLMCandidateEvaluator(
        crn_template=ToyCRN(),
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
        self.assertIn("RPA task + Search constraints + Feedback memory + Forbidden topology archive -> Decider", edges)
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
        trainer.configure_llm_graph(
            graph,
            every=2,
            task_description="Minimize toy loss.",
            num_candidates=10,
        )

        self.assertIsNone(trainer._maybe_run_llm_graph(1))
        result = trainer._maybe_run_llm_graph(2)

        self.assertIsNotNone(result)
        self.assertEqual(len(trainer.s.mult_env.hall_of_fame), 1)
        self.assertEqual(trainer.s.mult_env.hall_of_fame[0].state.last_task_info["source"], "LLM")
        self.assertEqual(trainer.llm_graph_history()[0]["requested"], 10)
        metric_names = {name for name, _, _ in logger.metrics}
        self.assertIn("LLM/Requested Count", metric_names)
        self.assertIn("LLM/Hall of Fame Size After", metric_names)
        self.assertIn("LLM/Timing Trainer Hook Seconds", metric_names)

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


if __name__ == "__main__":
    unittest.main()
