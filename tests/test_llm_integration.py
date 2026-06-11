import json
import tempfile
import unittest
from dataclasses import dataclass

import numpy as np

from RL4CRN.agent2env_interface.iocrn_stepper import IOCRNStepper
from RL4CRN.agent2env_interface.library_actuator import LibraryActuator
from RL4CRN.llm import (
    LLMCandidate,
    LLMCandidateEvaluator,
    LLMCRNGenerator,
    parse_candidates_payload,
)
from RL4CRN.utils.hall_of_fame import HallOfFame


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

        self.assertFalse(bad_id.valid)
        self.assertIn("outside the library", bad_id.message)
        self.assertFalse(bad_count.valid)
        self.assertIn("expected 2", bad_count.message)
        self.assertFalse(bad_params.valid)
        self.assertIn("expects 2 parameters", bad_params.message)

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


if __name__ == "__main__":
    unittest.main()
