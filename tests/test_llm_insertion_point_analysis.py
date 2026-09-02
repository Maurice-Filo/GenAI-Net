import importlib.util
import sqlite3
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "comparisons/rpa_search/scripts/plot_llm_insertion_point_analysis.py"
SPEC = importlib.util.spec_from_file_location("plot_llm_insertion_point_analysis", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class InsertionPointAnalysisTest(unittest.TestCase):
    def test_collect_database_tracks_insertion_and_missing_round(self):
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "results.sqlite"
            with sqlite3.connect(database) as connection:
                connection.executescript(
                    """
                    CREATE TABLE llm_runs (
                        llm_run_id TEXT, launched_epoch INTEGER, completed_epoch INTEGER,
                        requested INTEGER, produced INTEGER, valid_count INTEGER,
                        elapsed_seconds REAL
                    );
                    CREATE TABLE llm_candidates (
                        llm_run_id TEXT, topology_hash TEXT, valid INTEGER, loss REAL
                    );
                    CREATE TABLE evaluations (
                        topology_hash TEXT, parameters_json TEXT, source TEXT,
                        valid INTEGER, metadata_json TEXT
                    );
                    CREATE TABLE hof_snapshots (snapshot_id TEXT, epoch INTEGER);
                    CREATE TABLE hof_snapshot_entries (
                        snapshot_id TEXT, rank INTEGER, topology_hash TEXT, loss REAL,
                        parameters_json TEXT
                    );
                    INSERT INTO llm_runs VALUES ('round-0', 0, 2, 10, 1, 1, 12.0);
                    INSERT INTO llm_candidates VALUES ('round-0', 'llm-a', 1, 0.5);
                    INSERT INTO evaluations VALUES
                        ('llm-a', '[0.5]', 'llm', 1, '{"llm_run_id":"round-0"}');
                    INSERT INTO hof_snapshots VALUES ('s1', 1), ('s2', 2), ('s10', 10);
                    INSERT INTO hof_snapshot_entries VALUES
                        ('s1', 0, 'rl-a', 1.0, '[1.0]'),
                        ('s2', 0, 'llm-a', 0.5, '[0.5]'),
                        ('s10', 0, 'llm-a', 0.5, '[0.5]');
                    """
                )
            rows = MODULE.collect_database(
                database,
                task="logic",
                seed=0,
                expected_launches=[0, 5],
                launched_epochs={0, 5},
            )

        completed, missing = rows
        self.assertEqual(completed["epoch_lag"], 2)
        self.assertEqual(completed["post_insertion_epochs"], 8)
        self.assertEqual(completed["batch_to_pre_hof_ratio"], 0.5)
        self.assertEqual(completed["improved_rank1_at_insertion"], 1)
        self.assertEqual(completed["survived_in_final_hof"], 1)
        self.assertEqual(missing["status"], "launched_not_served")
        self.assertEqual(completed["launched"], 1)
        self.assertEqual(completed["served"], 1)
        self.assertEqual(missing["launched"], 1)
        self.assertEqual(missing["served"], 0)
        self.assertEqual(missing["entered_hof_at_insertion"], 0)


if __name__ == "__main__":
    unittest.main()
