import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from RL4CRN.llm.harness_client import (
    HarnessLLMClient,
    HarnessResponseError,
    _loads_json_response,
    build_crn_output_contract,
)
from RL4CRN.llm.deepseek_direct import _direct_prompt
from RL4CRN.llm.harness_runner import (
    HarnessCRNGenerator,
    HarnessDeciderWriterCRNGraph,
    _single_request_prompt,
)
from RL4CRN.llm.schemas import CandidateEvaluation
from RL4CRN.llm.workspace_tools import (
    WorkspaceEvaluationService,
    WorkspaceLiteratureService,
    default_workspace_tool_files,
)


class ToyReaction:
    def __init__(self, reaction_id, parameter_count):
        self.ID = reaction_id
        self.num_parameters = parameter_count

    def __str__(self):
        return f"R{self.ID}"


class ToyLibrary:
    def __init__(self):
        self.reactions = [ToyReaction(0, 1), ToyReaction(1, 2)]

    def __len__(self):
        return len(self.reactions)

    def get_reaction(self, index):
        return self.reactions[index]


def toy_contract(candidate_count=1):
    evaluator = SimpleNamespace(
        library=ToyLibrary(),
        max_added_reactions=2,
        require_unique_reactions=True,
    )
    return build_crn_output_contract(
        evaluator,
        num_candidates=candidate_count,
        task_description="Toy task",
    )


def bounded_toy_contract():
    evaluator = SimpleNamespace(
        library=ToyLibrary(),
        max_added_reactions=1,
        require_unique_reactions=True,
        min_parameter_value=0.001,
        max_parameter_value=100.0,
        enforce_parameter_bounds=True,
        forbidden_reaction_ids=[0],
    )
    return build_crn_output_contract(
        evaluator,
        num_candidates=1,
        task_description="Bounded toy task",
    )


class HarnessLLMClientTests(unittest.TestCase):
    def test_direct_prompt_preserves_hof_crn_when_raw_actions_are_unavailable(self):
        class HallState:
            last_task_info = {"reward": 0.25}

            def __str__(self):
                return "X_1 ----> X_2; [MAK(2.0)]"

        state = HallState()
        hall = [SimpleNamespace(state=state)]

        prompt = _direct_prompt(
            task_description="Toy task",
            reaction_library=ToyLibrary(),
            max_added_reactions=2,
            num_candidates=1,
            hall_of_fame_iter=hall,
        )

        self.assertIn('"loss": 0.25', prompt)
        self.assertIn('"crn": "X_1 ----> X_2; [MAK(2.0)]"', prompt)

    def test_single_request_prompt_requests_whole_batch_without_library_duplication(self):
        prompt = _single_request_prompt(10)

        self.assertIn("exactly 10 distinct CRN candidates", prompt)
        self.assertIn("6 new reaction-ID sets and 4 parameter refinements", prompt)
        self.assertIn("OUTPUT_GUIDE.json", prompt)
        self.assertIn("REACTION_LIBRARY.tsv", prompt)
        self.assertIn("Prefer proposing now", prompt)
        self.assertIn("next scheduled call", prompt)
        self.assertLess(len(prompt), 1200)
        self.assertNotIn("ID 0:", prompt)

    def test_headless_task_forbids_bash_before_workspace_reads(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            client = HarnessLLMClient(
                workspace_root=temp_dir,
                dsh_home=f"{temp_dir}/dsh-home",
                command=["fake-dsh"],
            )
            self.assertIn("Do not invoke Bash", client.system_prompt)
            self.assertIn("targeted grep queries", client.system_prompt)
            self.assertIn("at most eight workspace-tool calls", client.system_prompt)
            self.assertIn("The Decider owns the scientific choices", client.system_prompt)
            self.assertIn("The Writer must preserve those choices", client.system_prompt)
            self.assertIn("the Decider must not read the JSON guide", client.system_prompt)
            self.assertIn("[0.001, 100]", client.system_prompt)

    def test_contract_v2_declares_member_validation_and_endpoint_clamping(self):
        contract = bounded_toy_contract()

        self.assertEqual(contract["contract_version"], 2)
        self.assertEqual(contract["proposal_space_contract_version"], 2)
        self.assertEqual(contract["rules"]["allowed_parameter_range"], [0.001, 100.0])
        self.assertEqual(contract["rules"]["out_of_range_parameter_policy"], "clamp")
        self.assertEqual(contract["rules"]["candidate_validation_policy"], "independent-members")

    def test_harness_generator_makes_one_model_request_for_many_candidates(self):
        payload = {
            "candidates": [
                {"reaction_ids": [0], "parameter_values": [[1.0]]},
                {"reaction_ids": [1], "parameter_values": [[2.0, 3.0]]},
            ]
        }

        class RecordingClient(HarnessLLMClient):
            request_count = 0

            def generate_json(self, prompt, *, generation_config=None):
                self.request_count += 1
                return payload

        class Evaluator:
            library = ToyLibrary()
            max_added_reactions = 1
            require_unique_reactions = True
            forbidden_reaction_ids = frozenset()
            enforce_parameter_bounds = False
            min_parameter_value = 1e-6
            max_parameter_value = None

            def evaluate_many(self, candidates, **kwargs):
                return [
                    CandidateEvaluation(candidate=candidate, valid=True, loss=float(index))
                    for index, candidate in enumerate(candidates)
                ]

        with tempfile.TemporaryDirectory() as temp_dir:
            client = RecordingClient(
                workspace_root=temp_dir,
                dsh_home=f"{temp_dir}/dsh-home",
                command=["fake-dsh"],
            )
            generator = HarnessCRNGenerator(
                client=client,
                evaluator=Evaluator(),
                max_workspace_evaluations=0,
            )
            result = generator.run_round(task_description="Toy task", num_candidates=2)

        self.assertEqual(client.request_count, 1)
        self.assertEqual(len(result.candidates), 2)
        self.assertEqual(len(result.evaluations), 2)

    def test_harness_decider_writer_uses_two_calls_in_one_workspace(self):
        decision = "Design 1: R0 at k=1 and R1 at k=(2,3)."
        payload = {
            "candidates": [
                {"reaction_ids": [0, 1], "parameter_values": [[1.0], [2.0, 3.0]]}
            ]
        }

        class Evaluator:
            library = ToyLibrary()
            max_added_reactions = 2
            require_unique_reactions = True
            forbidden_reaction_ids = frozenset()
            enforce_parameter_bounds = True
            min_parameter_value = 0.001
            max_parameter_value = 100.0

            def evaluate_many(self, candidates, **kwargs):
                return [
                    CandidateEvaluation(candidate=candidate, valid=True, loss=0.5)
                    for candidate in candidates
                ]

        responses = [
            subprocess.CompletedProcess(args=[], returncode=0, stdout=decision, stderr=""),
            subprocess.CompletedProcess(
                args=[], returncode=0, stdout=json.dumps(payload), stderr=""
            ),
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            client = HarnessLLMClient(
                workspace_root=temp_dir,
                dsh_home=f"{temp_dir}/dsh-home",
                command=["fake-dsh"],
                candidate_validation_policy="independent-members",
            )
            graph = HarnessDeciderWriterCRNGraph(
                client=client,
                evaluator=Evaluator(),
                writer_retry_limit=0,
                max_workspace_evaluations=0,
            )
            with patch.object(client, "_initialize_git"):
                with patch(
                    "RL4CRN.llm.harness_client.subprocess.run", side_effect=responses
                ) as run_process:
                    result = graph.run_round(task_description="Toy task", num_candidates=1)

            workspace = client.last_workspace
            self.assertIsNotNone(workspace)
            self.assertEqual(run_process.call_count, 2)
            self.assertEqual(workspace.call_count, 2)
            self.assertEqual(
                (workspace.path / "DECIDER_DESIGNS.md").read_text().strip(), decision
            )
            self.assertTrue((workspace.path / "calls/0001/request.md").is_file())
            self.assertTrue((workspace.path / "calls/0002/request.md").is_file())
            decider_request = (workspace.path / "calls/0001/request.md").read_text()
            writer_request = (workspace.path / "calls/0002/request.md").read_text()
            self.assertIn("CONTEXT/HALL_OF_FAME.md", decider_request)
            self.assertNotIn("No Hall-of-Fame entries are available yet", decider_request)
            self.assertIn("OUTPUT_GUIDE.json", writer_request)
            self.assertIn("REACTION_LIBRARY.tsv", writer_request)
            self.assertNotIn("=== Available Reaction Library ===", writer_request)
            self.assertNotIn(decision, writer_request)
            self.assertEqual(
                json.loads((workspace.path / "WRITER_PAYLOAD.json").read_text()), payload
            )
            self.assertEqual(len(result.evaluations), 1)
            self.assertEqual(result.response_validation["accepted_candidate_count"], 1)
            self.assertEqual(result.response_validation["provider_call_count"], 2)

    def test_workspace_context_skill_and_reasoning_notes_are_baselined(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            client = HarnessLLMClient(
                workspace_root=temp_dir,
                dsh_home=f"{temp_dir}/dsh-home",
                command=["fake-dsh"],
            )
            files = default_workspace_tool_files()
            files["CONTEXT/HALL_OF_FAME.json"] = {"entries": [{"rank": 0, "loss": 1.2}]}
            with client.run(
                task_description="Toy task",
                contract=toy_contract(),
                workspace_files=files,
            ) as workspace:
                pass

            self.assertTrue((workspace.path / ".dsh/skills/crn-simulation/SKILL.md").is_file())
            skill = (workspace.path / ".dsh/skills/crn-simulation/SKILL.md").read_text()
            self.assertIn("tool-requests/probe-batch-01.request.json", skill)
            self.assertIn("at most three", skill)
            self.assertIn("default is **do not run an exploratory simulation**", skill)
            self.assertNotIn("evaluate_candidate.py", skill)
            self.assertTrue((workspace.path / "REASONING_NOTES.md").is_file())
            context = json.loads(
                (workspace.path / "CONTEXT/HALL_OF_FAME.json").read_text(encoding="utf-8")
            )
            self.assertEqual(context["entries"][0]["loss"], 1.2)
            tracked = subprocess.run(
                ["git", "-C", str(workspace.path), "ls-files"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout
            self.assertIn("REASONING_NOTES.md", tracked)
            self.assertIn(".dsh/skills/crn-simulation/SKILL.md", tracked)

    def test_workspace_file_queue_returns_text_diagnostics_and_enforces_cap(self):
        class Evaluator:
            def evaluate(self, candidate):
                state = SimpleNamespace(last_task_info={}, __str__=lambda self: "toy CRN")
                env = SimpleNamespace(state=state)
                return CandidateEvaluation(
                    candidate=candidate,
                    valid=True,
                    loss=0.25,
                    env=env,
                    message="valid",
                    task_info={"outputs": [[[0.1, 0.2, 0.3]]]},
                )

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for name, content in default_workspace_tool_files().items():
                target = root / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(content, encoding="utf-8")
            with WorkspaceEvaluationService(root, Evaluator(), max_evaluations=1):
                batch = self._queue_evaluation(root, "probe-batch-01", candidate_count=2)
            first_payload, second_payload = batch["evaluations"]
            self.assertTrue(first_payload["valid"])
            self.assertEqual(first_payload["loss"], 0.25)
            self.assertTrue((root / first_payload["diagnostics"]).is_file())
            diagnostics = json.loads((root / first_payload["diagnostics"]).read_text())
            self.assertEqual(diagnostics["trajectories"][0]["terminal_values"], [0.3])
            self.assertIn("exhausted", second_payload["error"])
            self.assertEqual(json.loads((root / "tool_evaluation_summary.json").read_text())["used"], 1)

    def test_literature_queue_searches_fixed_read_only_corpus(self):
        database = Path(__file__).parents[1] / "literature_rag/index/literature.sqlite3"
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            with WorkspaceLiteratureService(root, database, max_searches=1):
                queue = root / "literature-requests"
                request = queue / "search-01.request.json"
                response = queue / "search-01.response.json"
                request.write_text(
                    json.dumps({"query": "robust perfect adaptation", "limit": 2}),
                    encoding="utf-8",
                )
                for _ in range(300):
                    if response.is_file():
                        payload = json.loads(response.read_text(encoding="utf-8"))
                        break
                    __import__("time").sleep(0.01)
                else:
                    self.fail("literature service did not respond")

            self.assertTrue(payload["valid"])
            self.assertLessEqual(len(payload["results"]), 2)
            self.assertEqual(payload["used"], 1)

    @staticmethod
    def _queue_evaluation(root, request_id, candidate_count=1):
        queue = root / "tool-requests"
        request = queue / f"{request_id}.request.json"
        response = queue / f"{request_id}.response.json"
        request.write_text(
            json.dumps(
                {
                    "candidates": [
                        {
                            "reaction_ids": [0],
                            "parameter_values": [[1.0]],
                        }
                        for _ in range(candidate_count)
                    ]
                }
            ),
            encoding="utf-8",
        )
        for _ in range(200):
            if response.is_file():
                return json.loads(response.read_text(encoding="utf-8"))
            __import__("time").sleep(0.01)
        raise AssertionError("workspace evaluator did not respond")

    def test_json_extraction_prefers_payload_object_over_prose_range(self):
        response = (
            "All parameters are in [0.1, 50.0].\n"
            "```json\n"
            '{"candidates": [{"reaction_ids": [0, 1], '
            '"parameter_values": [[1.0], [2.0, 3.0]]}]}\n'
            "```"
        )
        payload = _loads_json_response(response)
        self.assertEqual(payload["candidates"][0]["reaction_ids"], [0, 1])

    def test_run_is_isolated_logged_and_strips_secret_environment(self):
        payload = {
            "candidates": [
                {"reaction_ids": [0, 1], "parameter_values": [[1.0], [2.0, 3.0]]}
            ]
        }
        completed = subprocess.CompletedProcess(
            args=[], returncode=0, stdout=json.dumps(payload), stderr=""
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            client = HarnessLLMClient(
                workspace_root=temp_dir,
                dsh_home=f"{temp_dir}/dsh-home",
                command=["fake-dsh"],
            )
            with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "must-not-leak"}):
                with client.run(
                    task_description="Toy task",
                    contract=toy_contract(),
                ) as workspace:
                    with patch(
                        "RL4CRN.llm.harness_client.subprocess.run", return_value=completed
                    ) as run_process:
                        result = client.generate_json("Generate one candidate")

            self.assertEqual(result, payload)
            self.assertTrue((workspace.path / ".git").is_dir())
            self.assertEqual(
                subprocess.run(
                    ["git", "-C", str(workspace.path), "log", "-1", "--format=%s"],
                    capture_output=True,
                    text=True,
                    check=True,
                ).stdout.strip(),
                "Initialize CRN Harness run",
            )
            self.assertTrue((workspace.path / "OUTPUT_CONTRACT.json").is_file())
            self.assertTrue((workspace.path / "OUTPUT_GUIDE.json").is_file())
            reaction_rows = (workspace.path / "REACTION_LIBRARY.tsv").read_text(
                encoding="utf-8"
            )
            self.assertIn("id\tparameter_count\tdisplay", reaction_rows)
            self.assertTrue((workspace.path / "SYSTEM_PROMPT.md").is_file())
            profile_patch = (workspace.path / "harness.patch.yml").read_text(encoding="utf-8")
            self.assertIn('model: "deepseek-v4-flash"', profile_patch)
            self.assertIn("id: system-prompt", profile_patch)
            self.assertTrue((workspace.path / "calls/0001/stdout.txt").is_file())
            self.assertTrue((workspace.path / "run_status.json").is_file())
            kwargs = run_process.call_args.kwargs
            self.assertEqual(kwargs["cwd"], workspace.path)
            self.assertFalse(kwargs.get("shell", False))
            self.assertIn("--patch", run_process.call_args.args[0])
            bot_task = run_process.call_args.args[0][-1]
            self.assertIn("This is the Writer call", bot_task)
            self.assertIn("DECIDER_DESIGNS.md", bot_task)
            self.assertIn("FINAL_RESPONSE.json", bot_task)
            self.assertNotIn("DEEPSEEK_API_KEY", kwargs["env"])
            self.assertEqual(kwargs["env"]["DSH_HOME"], str(Path(temp_dir, "dsh-home").resolve()))

    def test_decider_harness_call_is_free_form_and_not_given_json_contract(self):
        completed = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="Design 1: X_1 -> X_2 at k=2", stderr=""
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            client = HarnessLLMClient(
                workspace_root=temp_dir,
                dsh_home=f"{temp_dir}/dsh-home",
                command=["fake-dsh"],
            )
            with patch(
                "RL4CRN.llm.harness_client.subprocess.run", return_value=completed
            ) as run_process:
                with client.run(task_description="Toy task"):
                    decision = client.generate_text("Choose one design")

        bot_task = run_process.call_args.args[0][-1]
        self.assertIn("This is the Decider call", bot_task)
        self.assertIn("DECIDER_DESIGNS.md", bot_task)
        self.assertIn("do not emit machine JSON", bot_task)
        self.assertNotIn("FINAL_RESPONSE.json", bot_task)
        self.assertEqual(decision, completed.stdout)

    def test_request_uses_a_reusable_slot_profile_when_template_exists(self):
        completed = subprocess.CompletedProcess(
            args=[], returncode=0, stdout='{"answer": "ok"}', stderr=""
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            dsh_home = Path(temp_dir, "dsh-home")
            template = dsh_home / "profiles" / "headless"
            template.mkdir(parents=True)
            (template / "package.json").write_text(
                json.dumps(
                    {
                        "name": "dsh-profile-headless",
                        "private": True,
                        "dsh": {"profile": {"bundles": ["@deepseek-ai/dsh-headless"]}},
                    }
                ),
                encoding="utf-8",
            )
            (template / "cordis.patch.yml").write_text("[]\n", encoding="utf-8")
            (template / "pnpm-workspace.yaml").write_text("packages:\n  - .\n", encoding="utf-8")
            (template / "cordis.yml").write_text("[]\n", encoding="utf-8")
            client = HarnessLLMClient(
                workspace_root=temp_dir,
                dsh_home=dsh_home,
                command=["fake-dsh"],
            )

            with patch(
                "RL4CRN.llm.harness_client.subprocess.run", return_value=completed
            ) as run_process:
                with client.run(task_description="Toy task"):
                    client.generate_json("Return an object")

            command = run_process.call_args.args[0]
            profile_name = command[command.index("--profile") + 1]
            self.assertEqual(profile_name, "headless-worker-deepseek-official-000")
            profile_dir = dsh_home / "profiles" / profile_name
            self.assertEqual((profile_dir / "cordis.yml").read_text(), "[]\n")
            self.assertEqual(
                json.loads((profile_dir / "package.json").read_text())["name"],
                f"dsh-profile-{profile_name}",
            )

            with patch(
                "RL4CRN.llm.harness_client.subprocess.run", return_value=completed
            ) as second_process:
                with client.run(task_description="Another toy task"):
                    client.generate_json("Return another object")
            second_command = second_process.call_args.args[0]
            self.assertEqual(
                second_command[second_command.index("--profile") + 1], profile_name
            )

    def test_local_openai_route_is_scoped_to_loopback_and_written_to_patch(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            client = HarnessLLMClient(
                workspace_root=temp_dir,
                dsh_home=f"{temp_dir}/dsh-home",
                command=["fake-dsh"],
                provider="local-llama",
                model="qwen35-9b",
                openai_compatible_base_url="http://127.0.0.1:8080/v1/",
            )
            with client.run(task_description="Toy task") as workspace:
                profile_patch = (workspace.path / "harness.patch.yml").read_text(
                    encoding="utf-8"
                )

            self.assertIn("id: llm-pi-ai", profile_patch)
            self.assertIn('baseURL: "http://127.0.0.1:8080/v1"', profile_patch)
            self.assertIn("apiKeyEnv: DSH_LOCAL_LLM_API_KEY", profile_patch)
            self.assertIn("contextWindow: 32768", profile_patch)
            self.assertIn('provider: "local-llama"', profile_patch)
            self.assertEqual(
                client._child_environment()["DSH_LOCAL_LLM_API_KEY"],
                "local-loopback-no-auth",
            )
            self.assertIn('model: "qwen35-9b"', profile_patch)

    def test_local_openai_route_rejects_non_loopback_endpoint(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(ValueError, "localhost"):
                HarnessLLMClient(
                    workspace_root=temp_dir,
                    dsh_home=f"{temp_dir}/dsh-home",
                    provider="local-llama",
                    model="qwen35-9b",
                    openai_compatible_base_url="http://192.168.1.5:8080/v1",
                )

    def test_contract_rejects_wrong_parameter_arity(self):
        bad_payload = {
            "candidates": [
                {"reaction_ids": [0, 1], "parameter_values": [[1.0], [2.0]]}
            ]
        }
        completed = subprocess.CompletedProcess(
            args=[], returncode=0, stdout=json.dumps(bad_payload), stderr=""
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            client = HarnessLLMClient(
                workspace_root=temp_dir,
                dsh_home=f"{temp_dir}/dsh-home",
                command=["fake-dsh"],
            )
            with patch("RL4CRN.llm.harness_client.subprocess.run", return_value=completed):
                with client.run(task_description="Toy task", contract=toy_contract()):
                    with self.assertRaisesRegex(HarnessResponseError, "expects 2 parameters"):
                        client.generate_json("Generate one candidate")

    def test_contract_normalizes_unambiguous_scalar_parameter_shorthand(self):
        contract = {
            "required_candidate_count": 2,
            "reaction_library": [
                {"id": 10, "parameter_count": 1},
                {"id": 11, "parameter_count": 1},
            ],
            "rules": {
                "reaction_count_per_candidate": 2,
                "reaction_ids_must_be_unique": True,
            },
        }
        payload = {
            "candidates": [
                {"reaction_ids": [10, 11], "parameter_values": [0.5, 2.0]},
                {"reaction_ids": [10, 11], "parameter_values": [[0.7, 3.0]]},
            ]
        }
        completed = subprocess.CompletedProcess(
            args=[], returncode=0, stdout=json.dumps(payload), stderr=""
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            client = HarnessLLMClient(
                workspace_root=temp_dir,
                dsh_home=f"{temp_dir}/dsh-home",
                command=["fake-dsh"],
            )
            with patch("RL4CRN.llm.harness_client.subprocess.run", return_value=completed):
                with client.run(task_description="Scalar task", contract=contract) as workspace:
                    result = client.generate_json("Generate candidates")

            self.assertEqual(result["candidates"][0]["parameter_values"], [[0.5], [2.0]])
            self.assertEqual(result["candidates"][1]["parameter_values"], [[0.7], [3.0]])
            audit = json.loads(
                (workspace.path / "response_normalization.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                [entry["input_shape"] for entry in audit["candidates"]],
                ["flat_scalars", "single_packed_vector"],
            )

    def test_contract_rejects_extra_candidate_fields(self):
        payload = {
            "candidates": [
                {
                    "reaction_ids": [0, 1],
                    "parameter_values": [[1.0], [2.0, 3.0]],
                    "reasoning": "not part of the machine contract",
                }
            ]
        }
        completed = subprocess.CompletedProcess(
            args=[], returncode=0, stdout=json.dumps(payload), stderr=""
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            client = HarnessLLMClient(
                workspace_root=temp_dir,
                dsh_home=f"{temp_dir}/dsh-home",
                command=["fake-dsh"],
            )
            with patch("RL4CRN.llm.harness_client.subprocess.run", return_value=completed):
                with client.run(task_description="Toy task", contract=toy_contract()):
                    with self.assertRaisesRegex(HarnessResponseError, "must contain only"):
                        client.generate_json("Generate one candidate")

    def test_independent_member_policy_keeps_valid_subset_and_audits_rejections(self):
        payload = {
            "candidates": [
                {
                    "reaction_ids": [0, 1],
                    "parameter_values": [[1.0], [2.0, 3.0]],
                    "reasoning": "annotation outside the machine schema",
                },
                {
                    "reaction_ids": [0, 99],
                    "parameter_values": [[1.0], [2.0]],
                },
                {
                    "reaction_ids": [1, 0],
                    "parameter_values": [[4.0, 5.0], [6.0]],
                },
            ],
            "summary": "extra top-level annotation",
        }
        completed = subprocess.CompletedProcess(
            args=[], returncode=0, stdout=json.dumps(payload), stderr=""
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            client = HarnessLLMClient(
                workspace_root=temp_dir,
                dsh_home=f"{temp_dir}/dsh-home",
                command=["fake-dsh"],
                candidate_validation_policy="independent-members",
            )
            with patch("RL4CRN.llm.harness_client.subprocess.run", return_value=completed):
                with client.run(
                    task_description="Toy task", contract=toy_contract(candidate_count=3)
                ) as workspace:
                    result = client.generate_json("Generate three candidates")

            audit = json.loads(
                (workspace.path / "response_member_validation.json").read_text(encoding="utf-8")
            )

        self.assertEqual(len(result["candidates"]), 2)
        self.assertNotIn("reasoning", result["candidates"][0])
        self.assertEqual(audit["accepted_candidate_indices"], [0, 2])
        self.assertEqual(audit["rejected_candidates"][0]["candidate_index"], 1)
        self.assertEqual(audit["ignored_top_level_fields"], ["summary"])
        self.assertFalse(audit["scientific_values_modified"])

    def test_independent_member_policy_still_rejects_wholly_invalid_batch(self):
        payload = {
            "candidates": [
                {"reaction_ids": [0, 99], "parameter_values": [[1.0], [2.0]]}
            ]
        }
        completed = subprocess.CompletedProcess(
            args=[], returncode=0, stdout=json.dumps(payload), stderr=""
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            client = HarnessLLMClient(
                workspace_root=temp_dir,
                dsh_home=f"{temp_dir}/dsh-home",
                command=["fake-dsh"],
                candidate_validation_policy="independent-members",
            )
            with patch("RL4CRN.llm.harness_client.subprocess.run", return_value=completed):
                with client.run(task_description="Toy task", contract=toy_contract()) as workspace:
                    with self.assertRaisesRegex(HarnessResponseError, "no independently valid"):
                        client.generate_json("Generate one candidate")
            audit = json.loads(
                (workspace.path / "response_member_validation.json").read_text(encoding="utf-8")
            )
            self.assertEqual(audit["accepted_candidate_count"], 0)
            self.assertEqual(audit["rejected_candidates"][0]["candidate_index"], 0)

    def test_contract_rejects_forbidden_reaction_and_clamps_finite_rate(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            client = HarnessLLMClient(
                workspace_root=temp_dir,
                dsh_home=f"{temp_dir}/dsh-home",
                command=["fake-dsh"],
            )
            forbidden = {"candidates": [{"reaction_ids": [0], "parameter_values": [[1.0]]}]}
            completed = subprocess.CompletedProcess(
                args=[], returncode=0, stdout=json.dumps(forbidden), stderr=""
            )
            with patch("RL4CRN.llm.harness_client.subprocess.run", return_value=completed):
                with client.run(
                    task_description="Bounded toy task",
                    contract=bounded_toy_contract(),
                ):
                    with self.assertRaisesRegex(HarnessResponseError, "unknown reaction ID"):
                        client.generate_json("Generate one candidate")

            out_of_range = {
                "candidates": [
                    {"reaction_ids": [1], "parameter_values": [[100.1, 0.0001]]}
                ]
            }
            completed = subprocess.CompletedProcess(
                args=[], returncode=0, stdout=json.dumps(out_of_range), stderr=""
            )
            with patch("RL4CRN.llm.harness_client.subprocess.run", return_value=completed):
                with client.run(
                    task_description="Bounded toy task",
                    contract=bounded_toy_contract(),
                ) as workspace:
                    result = client.generate_json("Generate one candidate")
            self.assertEqual(result["candidates"][0]["parameter_values"], [[100.0, 0.001]])
            clamp_audit = json.loads(
                (workspace.path / "response_parameter_clamping.json").read_text(encoding="utf-8")
            )
            self.assertEqual(clamp_audit["clamped_parameter_count"], 2)


if __name__ == "__main__":
    unittest.main()
