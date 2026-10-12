from comparisons.llm_seeded_evolution.parity12.trial.campaign import prefix_events


def test_prefix_does_not_pull_late_replies_forward():
    events = [{"after_search_evaluations": n} for n in [1023, 102300, 103323, 306900]]
    assert prefix_events(events, 102300) == events[:2]
    assert events[-1]["after_search_evaluations"] == 306900
