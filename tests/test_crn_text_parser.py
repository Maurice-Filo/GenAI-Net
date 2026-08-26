from RL4CRN.utils.crn_text_parser import reactions_from_text
from RL4CRN.iocrns.reactions import CatalyticMichaelisMenten, MassAction


EXAMPLE = """
CRN hof_0, Reward: 0.1909286592422015
Inputs: ['u_1', 'u_2', 'u_3']
Species: ['X_1', 'X_2', 'X_3', 'X_4', 'X_5', 'X_6']
Output Species: ['X_6']
∅ ----> X_1;  [MAK(1.0, u_1)]
∅ ----> X_2;  [MAK(1.0, u_2)]
∅ ----> X_3;  [MAK(1.0, u_3)]
X_1 ----> X_3 + X_6;  [MAK(0.39422205090522766)]
X_2 ----> X_6 + X_6;  [MAK(0.4535057842731476)]
X_3 ----> X_6;  [MAK(0.4079498052597046)]
X_3 ----> X_4 + X_4;  [MAK(0.4053219258785248)]
X_1 + X_6 ----> X_1;  [MAK(0.40311625599861145)]
X_1 + X_6 ----> X_1 + X_5;  [MAK(0.4166227877140045)]
X_2 + X_3 ----> X_6 + X_6;  [MAK(0.37525293231010437)]
X_2 + X_4 ----> X_6;  [MAK(0.4266620874404907)]
X_2 + X_4 ----> X_6 + X_6;  [MAK(0.3628200590610504)]
X_3 + X_3 ----> X_6;  [MAK(0.4321376383304596)]
"""


def test_reactions_from_text_parses_printed_iocrn_reactions():
    reactions = reactions_from_text(EXAMPLE)

    assert len(reactions) == 13

    assert reactions[0].reactant_labels == []
    assert reactions[0].product_labels == ["X_1"]
    assert reactions[0].input_channels == ["u_1"]
    assert reactions[0].rate_constant == 1.0

    assert reactions[4].reactant_labels == ["X_2"]
    assert reactions[4].product_labels == ["X_6", "X_6"]
    assert reactions[4].input_channels == [None]
    assert reactions[4].rate_constant == 0.4535057842731476

    assert reactions[-1].reactant_labels == ["X_3", "X_3"]
    assert reactions[-1].product_labels == ["X_6"]
    assert reactions[-1].rate_constant == 0.4321376383304596


def test_reactions_from_text_parses_catalytic_michaelis_menten_reactions():
    text = """
    Inputs: ['u_1']
    Species: ['X_1', 'X_2', 'X_3']
    ∅ ----> X_1;  [MAK(1.0, u_1)]
    X_1 ----> X_2;  [CatalyticMM(E=X_3, k=0.25, K=1.5e-03)]
    """

    reactions = reactions_from_text(text)

    assert len(reactions) == 2
    assert isinstance(reactions[0], MassAction)
    assert isinstance(reactions[1], CatalyticMichaelisMenten)
    assert reactions[1].substrate_label == ["X_1"]
    assert reactions[1].product_label == ["X_2"]
    assert reactions[1].catalyst_label == ["X_3"]
    assert reactions[1].maximal_rate == 0.25
    assert reactions[1].michaelis_constant == 1.5e-03
