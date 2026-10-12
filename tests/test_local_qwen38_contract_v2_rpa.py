import unittest

from comparisons.llm_crn_generation.run_local_qwen38_contract_v2_rpa import (
    CAMPAIGN_ID,
    ENDPOINT,
    campaign_command,
    tool_probe_payload,
)


class LocalQwen38ContractV2RpaTests(unittest.TestCase):
    def test_campaign_matches_the_flash_rpa_protocol(self):
        command = campaign_command("Qwen3.8-test")
        joined = " ".join(command)

        expected = (
            "--tasks rpa",
            "--seeds 1",
            "--seed-start 9000",
            "--epochs 100",
            "--rl-batch-size 1023",
            "--total-candidate-budget 102400",
            "--llm-candidates 10",
            "--llm-every 20",
            "--max-llm-in-flight 5",
            "--generation-backend harness",
            "--communication-mode full",
            "--withhold-initial-hof",
            "--candidate-validation-policy independent-members",
            "--campaign-stage sentinel",
            "--task-prompt-variant standard",
            "--max-agent-evaluations 0",
        )
        for fragment in expected:
            self.assertIn(fragment, joined)

    def test_local_endpoint_and_model_are_forwarded_verbatim(self):
        command = campaign_command("Qwen3.8-exact-id")

        self.assertEqual(ENDPOINT, "http://127.0.0.1:18082/v1")
        self.assertTrue(CAMPAIGN_ID.endswith("-r8"))
        self.assertEqual(command[command.index("--llm-base-url") + 1], ENDPOINT)
        self.assertEqual(
            command[command.index("--local-llm-context-window") + 1], "262144"
        )
        self.assertEqual(
            command[command.index("--local-llm-max-output-tokens") + 1], "8192"
        )
        self.assertEqual(command[command.index("--model") + 1], "Qwen3.8-exact-id")
        self.assertTrue(
            command[command.index("--method-name") + 1].endswith("_r8")
        )
        self.assertTrue(command[command.index("--run-suffix") + 1].endswith("_r8"))

    def test_preflight_requires_an_automatic_tool_call(self):
        payload = tool_probe_payload("Qwen3.8-exact-id")

        self.assertEqual(payload["model"], "Qwen3.8-exact-id")
        self.assertEqual(payload["tool_choice"], "auto")
        self.assertEqual(payload["chat_template_kwargs"], {"enable_thinking": True})
        self.assertEqual(payload["max_tokens"], 1024)
        self.assertEqual(payload["thinking_token_budget"], 256)
        self.assertEqual(payload["tools"][0]["function"]["name"], "noop")
        self.assertFalse(
            payload["tools"][0]["function"]["parameters"]["additionalProperties"]
        )


if __name__ == "__main__":
    unittest.main()
