#!/usr/bin/env python3
"""Optional paper comparison of No communication and LLM-seeded evolution."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from comparisons.llm_seeded_evolution.compare import main

if __name__ == "__main__":
    main()
