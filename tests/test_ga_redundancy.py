from comparisons.llm_seeded_evolution.analysis.audit_ga_redundancy import eligible, is_null


def test_net_zero_is_not_just_empty_reaction():
    class Reaction:
        def __init__(self, net):
            self.net = net

        def get_stoichiometry_dict(self):
            return self.net

    assert is_null(Reaction({}))
    assert is_null(Reaction({"X": 0}))
    assert not is_null(Reaction({"X": -1, "Y": 1}))


def test_deletion_requires_trajectory_loss_and_truth_table():
    row = {"status": "ok", "all16_correct": True, "loss": .02, "max_output_deviation": .009}
    assert eligible(row, .01, .04)
    assert not eligible(row, .001, .04)
    assert not eligible(row | {"all16_correct": False}, .01, .04)
    assert not eligible(row | {"loss": .04}, .01, .04)
    assert not eligible(row | {"status": "solver_failure"}, .01, .04)
