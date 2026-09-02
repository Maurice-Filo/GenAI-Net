import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "comparisons/llm_crn_generation/run_genai_net_llm_20seed_campaign.py"
FLASH_QUEUE = ROOT / "comparisons/llm_crn_generation/run_flash_paper_campaign_queue.sh"
LOCAL_QUEUE = ROOT / "comparisons/llm_crn_generation/run_local_qwen_campaign_queue.sh"
CLASSIFIER_ABLATION = (
    ROOT
    / "comparisons/llm_crn_generation/run_classifier_communication_ablation_queue.sh"
)
LOGIC_PROMPT_ABLATION = (
    ROOT / "comparisons/llm_crn_generation/run_logic_trajectory_prompt_campaign.sh"
)


class CampaignGpuAssignmentTests(unittest.TestCase):
    def test_launcher_exposes_selected_physical_gpu_to_children(self):
        tree = ast.parse(LAUNCHER.read_text(encoding="utf-8"))
        assignments = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Subscript)
                and ast.unparse(target) == "env['CUDA_VISIBLE_DEVICES']"
                for target in node.targets
            )
        ]

        self.assertEqual(len(assignments), 1)
        self.assertEqual(ast.unparse(assignments[0].value), "str(args.rl_gpu)")

    def test_api_queues_default_to_better_gpu_and_local_queue_preserves_gpu_split(self):
        lookup = 'nvidia-smi --id="$RL_GPU_INDEX" --query-gpu=uuid'
        self.assertGreaterEqual(
            FLASH_QUEUE.read_text(encoding="utf-8").count('--rl-gpu "$RL_GPU_UUID"'),
            5,
        )
        self.assertIn(lookup, FLASH_QUEUE.read_text(encoding="utf-8"))
        self.assertIn('RL_GPU_INDEX="${RL_GPU_INDEX:-1}"', FLASH_QUEUE.read_text(encoding="utf-8"))
        local_queue = LOCAL_QUEUE.read_text(encoding="utf-8")
        self.assertIn("nvidia-smi --id=0 --query-gpu=uuid", local_queue)
        self.assertIn('--rl-gpu "$RL_GPU_UUID"', local_queue)
        self.assertIn("CUDA_DEVICE_ORDER=PCI_BUS_ID", local_queue)

    def test_campaign_shutdown_terminates_worker_process_groups(self):
        tree = ast.parse(LAUNCHER.read_text(encoding="utf-8"))
        calls = [ast.unparse(node.func) for node in ast.walk(tree) if isinstance(node, ast.Call)]

        self.assertIn("os.killpg", calls)
        self.assertIn("terminate_process_groups", calls)

    def test_campaign_forwards_initial_hof_ablation(self):
        source = LAUNCHER.read_text(encoding="utf-8")

        self.assertIn('cmd.append("--withhold-initial-hof")', source)
        self.assertIn('"--task-prompt-variant"', source)
        self.assertIn('"task_prompt_sha256_by_task": prompt_hashes', source)
        self.assertIn("validate_experiment_release(", source)
        self.assertIn('"--prompt-approval-file"', source)
        self.assertIn('"--analysis-plan-file"', source)
        self.assertIn('"--campaign-stage"', source)

    def test_flash_queue_contains_matched_no_communication_ablation(self):
        queue = FLASH_QUEUE.read_text(encoding="utf-8")
        self.assertIn("flash-no-communication-long300-20seed", queue)
        self.assertIn("--communication-mode none", queue)
        self.assertIn("300 1023 307200", queue)

    def test_classifier_communication_ablation_is_matched_and_isolated(self):
        queue = CLASSIFIER_ABLATION.read_text(encoding="utf-8")

        self.assertIn("--tasks classifier", queue)
        self.assertIn("--epochs 100", queue)
        self.assertIn("--rl-batch-size 1023", queue)
        self.assertIn("--total-candidate-budget 102400", queue)
        self.assertIn("--communication-mode none", queue)
        self.assertIn("--max-parallel 12", queue)
        self.assertIn('RL_GPU_INDEX="${RL_GPU_INDEX:-0}"', queue)

    def test_logic_prompt_ablation_is_explicit_and_matched(self):
        queue = LOGIC_PROMPT_ABLATION.read_text(encoding="utf-8")

        self.assertIn("--campaign-id \"$CAMPAIGN_ID\"", queue)
        self.assertIn("--tasks logic", queue)
        self.assertIn("--seeds 20", queue)
        self.assertIn("--epochs 100", queue)
        self.assertIn("--rl-batch-size 1023", queue)
        self.assertIn("--total-candidate-budget 102400", queue)
        self.assertIn("--task-prompt-variant logic-trajectory", queue)
        self.assertIn("--communication-mode full", queue)
        self.assertIn("--withhold-initial-hof", queue)
        self.assertIn("--max-agent-evaluations 0", queue)
        self.assertIn("--max-parallel 10", queue)
        self.assertIn('RL_GPU_INDEX="${RL_GPU_INDEX:-1}"', queue)
        self.assertIn("genai_net_llm_flash_logic_trajectory_prompt", queue)
        self.assertIn("cvode_llm_flash_logic_trajectory_prompt", queue)


if __name__ == "__main__":
    unittest.main()
