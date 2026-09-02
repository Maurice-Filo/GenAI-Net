import importlib.util
import sys
import sqlite3
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).parents[1]
SPEC = importlib.util.spec_from_file_location("fetch_papers", ROOT / "fetch_papers.py")
FETCH = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = FETCH
SPEC.loader.exec_module(FETCH)

SEARCH_SPEC = importlib.util.spec_from_file_location("search", ROOT / "search.py")
SEARCH = importlib.util.module_from_spec(SEARCH_SPEC)
assert SEARCH_SPEC.loader is not None
sys.modules[SEARCH_SPEC.name] = SEARCH
SEARCH_SPEC.loader.exec_module(SEARCH)


class LiteratureRagTests(unittest.TestCase):
    def test_extract_jats_keeps_abstract_and_direct_section_paragraphs(self):
        xml = b"""<article><front><article-title>A test</article-title>
          <abstract><p>Short abstract.</p></abstract></front><body>
          <sec><title>Results</title><p>Useful result.</p>
            <sec><title>Nested</title><p>Nested result.</p></sec>
          </sec></body></article>"""
        title, text = FETCH.extract_jats(xml)
        self.assertEqual(title, "A test")
        self.assertIn("ABSTRACT\nShort abstract.", text)
        self.assertIn("RESULTS\nUseful result.", text)
        self.assertIn("NESTED\nNested result.", text)

    def test_index_has_fts5(self):
        with tempfile.TemporaryDirectory() as directory:
            connection = FETCH.connect_index(Path(directory) / "index.sqlite3")
            connection.execute(
                "INSERT INTO paper_text (pmcid, title, body) VALUES (?, ?, ?)",
                ("PMC1", "Integral feedback", "robust perfect adaptation"),
            )
            count = connection.execute(
                "SELECT COUNT(*) FROM paper_text WHERE paper_text MATCH 'adaptation'"
            ).fetchone()[0]
            self.assertEqual(count, 1)
            connection.close()

    def test_search_database_opens_index_read_only(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "index.sqlite3"
            connection = FETCH.connect_index(path)
            connection.execute(
                """INSERT INTO papers
                   (pmcid, title, source_url, license_status, xml_path, text_path, sha256)
                   VALUES ('PMC1', 'Integral feedback', 'https://example.test', 'OA', 'a', 'b', 'c')"""
            )
            connection.execute(
                "INSERT INTO paper_text (pmcid, title, body) VALUES ('PMC1', 'Integral feedback', 'robust perfect adaptation')"
            )
            connection.commit()
            connection.close()
            rows = SEARCH.search_database(path, "robust AND adaptation")
            self.assertEqual(rows[0]["pmcid"], "PMC1")
            with self.assertRaises(sqlite3.OperationalError):
                readonly = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
                try:
                    readonly.execute("DELETE FROM papers")
                finally:
                    readonly.close()


if __name__ == "__main__":
    unittest.main()
