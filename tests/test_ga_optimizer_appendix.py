import numpy as np
import pytest
from scipy.stats import PermutationMethod, wilcoxon

from paper.iclr2027_genai_net_llm.generate_ga_optimizer_appendix import (
    exact_pratt_pvalue, perturbation_contrasts,
)


@pytest.mark.parametrize("d",[[1,2,2,-1,0],[0,0,1,1,-2],[1,-1,1,-1],[1,2,3,4,5]])
def test_exact_pratt_matches_scipy_permutations(d):
    reference=wilcoxon(d,zero_method="pratt",method=PermutationMethod(n_resamples=np.inf)).pvalue
    assert exact_pratt_pvalue(d)==reference


def test_zeros_and_sign_symmetry():
    assert exact_pratt_pvalue([0,0,0])==1
    assert exact_pratt_pvalue([1]*20)==2/(2**20)
    d=np.array([0,1,2,-3,3,-1])
    assert exact_pratt_pvalue(d)==exact_pratt_pvalue(-d)


def test_complete_family_and_missing_seed_rejection():
    rows=[{"arm":arm,"seed":seed,"mode":"all_rates","radius":r,"draws":32,
        "qualified_fraction":v} for arm,v in (("julia",1.),("evolution",.5),("ga",0.))
        for seed in range(1,21) for r in (1.01,1.05,1.1,1.25,1.5)]
    contrasts=perturbation_contrasts(rows)
    assert len(contrasts)==15
    assert all(c["n"]==20 and c["wins"]==20 for c in contrasts)
    assert all(c["p_holm_15"]==15*2/(2**20) for c in contrasts)
    with pytest.raises(ValueError,match="20 matched"):
        perturbation_contrasts(rows[:-1])
    with pytest.raises(ValueError,match="20 matched"):
        perturbation_contrasts(rows+[rows[0]])
