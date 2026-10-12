from comparisons.search_exploration.plot_logic5_canonical_topology_counts import canonical_hash, count_memberships


def reaction(reactants, products, inputs=None, rate=1., identifier=1):
    return {"reaction_id": identifier, "parameters": [rate], "structure": {
        "type": "MassAction", "reactants": reactants, "products": products,
        "inputs": [None] if inputs is None else inputs}}


def test_permutations_rates_and_library_aliases_do_not_change_topology():
    a = reaction(["A", "B"], ["C"], rate=3.)
    b = reaction(["C"], [], identifier=2)
    assert canonical_hash([a, b]) == canonical_hash([
        reaction(["C"], [], rate=99., identifier=200),
        reaction(["B", "A"], ["C"], rate=.001, identifier=100)])


def test_input_channels_multiplicities_and_species_roles_remain_distinct():
    assert canonical_hash([reaction([], ["A"], ["u_1"])]) != canonical_hash([reaction([], ["A"])])
    assert canonical_hash([reaction(["A", "A"], [])]) != canonical_hash([reaction(["A"], [])])
    a = reaction(["A"], ["B"])
    assert canonical_hash([a, a]) != canonical_hash([a])
    assert canonical_hash([a]) != canonical_hash([reaction(["B"], ["A"])])


def test_per_seed_and_pooled_counts_deduplicate_rates_aliases_and_recurrence():
    memberships = [{"arm": "full", "seed": seed, "topology_hash": raw, "core_hash": core,
                    "core_pass_fraction": fraction}
                   for seed, raw, core, fraction in [(1,"raw1","core1",.8), (1,"raw2","core2",1.),
                                                      (2,"raw1","core1",1.)]]
    rows, pooled = count_memberships(memberships, {"raw1":"same", "raw2":"same"},
                                     {"core1":"samecore", "core2":"samecore"}, .9)
    one = next(r for r in rows if r["arm"] == "full" and r["seed"] == 1)
    assert one["valid"] == one["cores"] == one["robust_cores"] == 1
    combined = next(r for r in pooled if r["arm"] == "full")
    assert combined["valid"] == combined["cores"] == combined["robust_cores"] == 1
    assert sum(r["cores"] for r in rows if r["arm"] == "full") == 2
