import json
import unittest

from tools.qwen_reasoning_budget_proxy import inject_reasoning_budget


class QwenReasoningBudgetProxyTests(unittest.TestCase):
    def test_adds_budget_to_reasoning_chat_request(self):
        payload = {"chat_template_kwargs": {"enable_thinking": True}}
        result = inject_reasoning_budget(
            "/v1/chat/completions", json.dumps(payload).encode(), 4096
        )
        self.assertEqual(json.loads(result)["thinking_token_budget"], 4096)

    def test_preserves_explicit_smaller_budget(self):
        payload = {"thinking_token_budget": 256}
        result = inject_reasoning_budget(
            "/v1/chat/completions", json.dumps(payload).encode(), 4096
        )
        self.assertEqual(json.loads(result)["thinking_token_budget"], 256)

    def test_does_not_budget_nonthinking_request(self):
        payload = {"chat_template_kwargs": {"enable_thinking": False}}
        body = json.dumps(payload).encode()
        self.assertEqual(inject_reasoning_budget("/v1/chat/completions", body, 4096), body)

    def test_does_not_modify_other_endpoints(self):
        body = b'{"data":[]}'
        self.assertEqual(inject_reasoning_budget("/v1/models", body, 4096), body)


if __name__ == "__main__":
    unittest.main()

