import unittest
from unittest.mock import MagicMock, patch

from comparisons.llm_crn_generation import run_genai_net_llm_20seed_campaign as campaign


class CometPreflightTests(unittest.TestCase):
    @patch.object(campaign, "Experiment")
    @patch.object(campaign, "API")
    def test_missing_authenticated_workspace_fails_before_experiment(
        self, api_class, experiment_class
    ):
        api_class.return_value.get_account_details.return_value = {}

        with self.assertRaisesRegex(RuntimeError, "account workspace"):
            campaign.comet_smoke("project", "campaign")

        experiment_class.assert_not_called()

    @patch.object(campaign, "Experiment")
    @patch.object(campaign, "API")
    def test_local_key_from_non_live_experiment_is_rejected(
        self, api_class, experiment_class
    ):
        api_class.return_value.get_account_details.return_value = {
            "defaultWorkspaceName": "workspace"
        }
        experiment = MagicMock(alive=False)
        experiment.get_key.return_value = "locally-allocated-key"
        experiment_class.return_value = experiment

        with self.assertRaisesRegex(RuntimeError, "not authenticated and live"):
            campaign.comet_smoke("project", "campaign")

        experiment.end.assert_called_once()

    @patch.object(campaign, "Experiment")
    @patch.object(campaign, "API")
    def test_authenticated_live_experiment_passes(self, api_class, experiment_class):
        api_class.return_value.get_account_details.return_value = {
            "defaultWorkspaceName": "workspace"
        }
        experiment = MagicMock(alive=True)
        experiment.get_key.return_value = "remote-key"
        experiment_class.return_value = experiment

        key = campaign.comet_smoke("project", "campaign")

        self.assertEqual(key, "remote-key")
        experiment.log_metric.assert_called_once_with(
            "preflight/comet_authenticated", 1, step=0
        )
        experiment.end.assert_called_once()


if __name__ == "__main__":
    unittest.main()
