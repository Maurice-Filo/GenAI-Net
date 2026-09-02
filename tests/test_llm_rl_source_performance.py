import importlib.util
import math
import sqlite3
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "comparisons/rpa_search/scripts/plot_llm_rl_source_performance.py"
SPEC = importlib.util.spec_from_file_location("plot_llm_rl_source_performance", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class SourcePerformanceTest(unittest.TestCase):
    def test_history_keeps_independent_source_minima(self):
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "results.sqlite"
            with sqlite3.connect(database) as connection:
                connection.executescript(
                    """
                    CREATE TABLE evaluations (
                        topology_hash TEXT, parameters_json TEXT, source TEXT, valid INTEGER
                    );
                    CREATE TABLE hof_snapshots (snapshot_id TEXT, epoch INTEGER);
                    CREATE TABLE hof_snapshot_entries (
                        snapshot_id TEXT, rank INTEGER, topology_hash TEXT,
                        parameters_json TEXT, loss REAL
                    );
                    INSERT INTO evaluations VALUES ('llm-a', '[2]', 'llm', 1);
                    INSERT INTO hof_snapshots VALUES ('s0', 0), ('s1', 1), ('s2', 2);
                    INSERT INTO hof_snapshot_entries VALUES
                        ('s0', 0, 'rl-a', '[1]', 1.0),
                        ('s1', 0, 'llm-a', '[2]', 0.8),
                        ('s1', 1, 'rl-b', '[3]', 0.9),
                        ('s2', 0, 'rl-c', '[4]', 0.7),
                        ('s2', 1, 'llm-a', '[2]', 0.8);
                    """
                )
            history = MODULE.read_run_history(database)

        self.assertEqual(history[0][0], 1.0)
        self.assertTrue(math.isnan(history[0][1]))
        self.assertEqual(history[1], (0.9, 0.8))
        self.assertEqual(history[2], (0.7, 0.8))


if __name__ == "__main__":
    unittest.main()
