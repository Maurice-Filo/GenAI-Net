"""Optional real-evaluator smoke test; outputs stay in pytest's temporary folder.

This checks mutation and simulation together, not performance at paper budget.
The first recorded proposal batch is made available at smoke-test step zero.
The production runner always retains the actual recorded completion schedule.
"""

from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path

import pytest

from comparisons.llm_seeded_evolution.run import evaluate, init_worker
from comparisons.llm_seeded_evolution.search import Evolution, run_search


@pytest.mark.skipif(not os.environ.get("GENAI_EVOLUTION_REPLAY"),
                    reason="Set GENAI_EVOLUTION_REPLAY to a real No communication replay.json.")
def test_real_proposals_and_both_mutations_with_cvode(tmp_path):
    replay = json.loads(Path(os.environ["GENAI_EVOLUTION_REPLAY"]).read_text())
    assert replay["task"] in {"logic", "rpa", "classifier"} and replay["communication_mode"] == "none"
    assert replay["expose_state_to_llm"] is False and replay["insert_llm_into_rl"] is False
    config = replay["scientific_config"]
    init_worker(config)
    from comparisons.llm_seeded_evolution import run
    evaluator = run._EVALUATOR
    allowed = [r.ID for r in evaluator.library.reactions if r.ID not in evaluator.forbidden_reaction_ids]
    n_reactions = config["search"]["max_added_reactions"]
    if replay["task"] == "classifier":
        assert n_reactions == 6 and evaluator.scenario_count == 32
        assert set(evaluator.forbidden_reaction_ids) == {0}
    evolution = Evolution(seed=7, allowed_ids=allowed, n_reactions=n_reactions, archive_size=30)
    event = {**replay["events"][0], "after_search_evaluations": 0}
    records = []
    with ProcessPoolExecutor(max_workers=2, initializer=init_worker, initargs=(config,)) as pool:
        counts = run_search(evolution, events=[event], search_budget=64, batch_size=8,
                            evaluate_batch=lambda cs: list(pool.map(evaluate, cs)),
                            record=lambda result, n, m: records.append((result, n, m)))
    operations = Counter(r.candidate.source for r, _, _ in records)
    assert counts == (64, len(event["candidates"]))
    assert operations["rate_mutation"] > 0 and operations["reaction_mutation"] > 0
    assert operations["random"] > 0
    initial_best = min(c["reference_loss"] for c in event["candidates"])
    assert evolution.archive.ranked()[0].loss <= initial_best
    assert all(len(r.candidate.reaction_ids) == n_reactions for r, _, _ in records)
    assert all(set(r.candidate.reaction_ids) <= set(allowed) for r, _, _ in records)
    report = {"functional_smoke_only": True, "schedule": "first_batch_at_zero_for_test_only",
              "source_branch": "none", "search_evaluations": 64,
              "llm_evaluations": len(event["candidates"]), "operations": dict(operations),
              "initial_best_loss": initial_best, "final_best_loss": evolution.archive.ranked()[0].loss}
    (tmp_path / f"{replay['task']}_smoke.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report))
