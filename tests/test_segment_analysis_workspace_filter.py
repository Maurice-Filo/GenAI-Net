import contextlib
import io
import tempfile
import unittest
from pathlib import Path

from comparisons.rpa_search.scripts.plot_llm_segment_analysis import initialized_workspaces


class SegmentAnalysisWorkspaceFilterTests(unittest.TestCase):
    def test_skips_output_only_workspace_from_orphaned_agent(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            valid = run / "harness-workspaces/20260821T-crn-generation-valid"
            valid.joinpath("CONTEXT").mkdir(parents=True)
            valid.joinpath("run_manifest.json").write_text("{}", encoding="utf-8")
            valid.joinpath("CONTEXT/SEARCH_STATE.json").write_text("{}", encoding="utf-8")

            orphan = run / "harness-workspaces/20260821T-crn-generation-orphan"
            orphan.mkdir(parents=True)
            orphan.joinpath("FINAL_RESPONSE.json").write_text("{}", encoding="utf-8")

            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                workspaces = initialized_workspaces(run)

            self.assertEqual(workspaces, [valid])
            self.assertIn("uninitialized Harness workspace", stderr.getvalue())
            self.assertIn("run_manifest.json", stderr.getvalue())
            self.assertIn("CONTEXT/SEARCH_STATE.json", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
