import numpy as np

from RL4CRN.iocrns.iocrn import IOCRN
from RL4CRN.iocrns.reactions import CatalyticMichaelisMenten


def test_catalytic_michaelis_menten_propensity_and_stoichiometry():
    reaction = CatalyticMichaelisMenten(
        substrate_label=["A"],
        product_label=["B"],
        catalyst_label=["E"],
        params=[2.0, 0.5],
        params_controllability=[False, False],
    )
    crn = IOCRN([reaction], output_labels=["B"], solver="LSODA")
    crn.compile()

    x = np.zeros(crn.num_species)
    x[crn.species_label_to_idx("A")] = 1.5
    x[crn.species_label_to_idx("E")] = 0.25

    expected_rate = 2.0 * 0.25 * 1.5 / (0.5 + 1.5)
    dx = crn.rate_function(0.0, x, np.asarray([]))

    assert np.isclose(dx[crn.species_label_to_idx("A")], -expected_rate)
    assert np.isclose(dx[crn.species_label_to_idx("B")], expected_rate)
    assert np.isclose(dx[crn.species_label_to_idx("E")], 0.0)
    assert np.isclose(dx.sum(), 0.0)


def test_catalytic_michaelis_menten_conserves_conversion_pool():
    reaction = CatalyticMichaelisMenten(
        substrate_label=["A"],
        product_label=["B"],
        catalyst_label=["E"],
        params=[3.0, 0.1],
        params_controllability=[False, False],
    )
    crn = IOCRN([reaction], output_labels=["B"], solver="LSODA", atol=1e-10, rtol=1e-8)
    crn.compile()

    x0 = np.zeros(crn.num_species)
    x0[crn.species_label_to_idx("A")] = 1.0
    x0[crn.species_label_to_idx("E")] = 0.4
    time = np.linspace(0.0, 10.0, 101)
    _, trajectories, _, _ = crn.transient_response(
        [np.asarray([])], [x0], time, force=True
    )
    x = trajectories[0]

    pool = x[crn.species_label_to_idx("A")] + x[crn.species_label_to_idx("B")]
    assert np.allclose(pool, 1.0, atol=1e-7)
    assert np.allclose(x[crn.species_label_to_idx("E")], 0.4, atol=1e-10)
