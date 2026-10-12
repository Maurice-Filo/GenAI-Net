from comparisons.search_exploration.analyze_logic5_robust_provenance import parameter_hash, route, merged_group


def test_parameter_hash_preserves_rate_alignment_and_ignores_order():
    selection = {"fixed": [], "catalog": {str(i): {"reaction_id": i, "structure": {
        "type": "MassAction", "reactants": [f"X{i}"], "products": [], "inputs": [None]}}
        for i in (1, 2)}}
    a = {"reaction_ids": [2, 1], "rates": [4., 3.]}
    b = {"reaction_ids": [1, 2], "rates": [3., 4.]}
    assert parameter_hash(a, selection) == parameter_hash(b, selection)
    assert parameter_hash(a, selection) != parameter_hash({**b, "rates": [4., 3.]}, selection)


def test_sampling_mutation_and_rate_refinement_are_distinct():
    assert route({"source": "rl", "origin": "rl", "stream": "search"}) == "rl_direct"
    assert route({"source": "rl", "origin": "rl", "stream": "optimization"}) == "rl_fit"
    assert route({"source": "llm", "origin": "llm"}) == "llm_direct"
    assert route({"source": "llm", "origin": "llm", "stream": "optimization"}) == "llm_fit"
    assert route({"source": "ga", "origin": "rl"}) == "ga_rl"
    assert route({"source": "ga", "origin": "llm", "stream": "optimization"}) == "ga_llm"
    assert route({"source": "random", "origin": "random"}) == "other"
    assert route({}) == "unknown"


def test_shared_routes_do_not_get_arbitrary_single_source_credit():
    assert merged_group(["llm_direct", "llm_direct"]) == "llm_direct"
    assert merged_group(["llm_direct", "rl_direct"]) == "multiple"
    assert merged_group(["rl_fit", "rl_direct"]) == "multiple"
    assert merged_group([]) == "unknown"
