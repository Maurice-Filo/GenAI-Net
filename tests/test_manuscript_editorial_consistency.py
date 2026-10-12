from pathlib import Path

import pytest


PAPER = Path(__file__).resolve().parents[1] / "paper/iclr2027_genai_net_llm"
SOURCES = [PAPER / "main.tex", *sorted((PAPER / "sections").glob("*.tex"))]


@pytest.mark.parametrize("source", SOURCES, ids=lambda p: p.name)
def test_narrative_does_not_reintroduce_removed_editorial_notes(source):
    text = " ".join(source.read_text().lower().split())
    for phrase in (
        "pending targeted reruns",
        "retained in the text",
        "reported here rather than plotted",
        "original frozen cohorts, not new runs",
        "final points include late replies",
        "with late replies confined to the final",
        "documents legacy runtime field names",
        "archived text is not rewritten",
    ):
        assert phrase not in text
