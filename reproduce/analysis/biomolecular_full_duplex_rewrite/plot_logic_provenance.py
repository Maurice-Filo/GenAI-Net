"""Render the manuscript provenance figure from the frozen analysis."""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

from comparisons.search_exploration.analyze_logic5_robust_provenance import render


def main():
    source = HERE.parent / "jmlr_hybrid_crn/generated/logic_provenance.json"
    render(
        HERE / "figures",
        json.loads(source.read_text()),
        title="Recorded paths to robust reduced Logic structures",
        include_footer=False,
        include_subtitle=False,
        stem="logic_provenance",
        structure_name="reduced structures",
    )


if __name__ == "__main__":
    main()
