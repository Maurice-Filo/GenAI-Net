import json

import pytest

from paper.iclr2027_genai_net_llm.generate_flash_logic_directional import (
    GENERATED, contrast, returned_batches,
)


def test_contrast_direction_and_ties():
    left, right = [1.0]*10+[2.0]*5+[3.0]*5, [2.0]*20
    loss = contrast(left, right, higher=False)
    count = contrast(left, right, higher=True)
    assert (loss["wins"], loss["ties"], loss["losses"]) == (10, 5, 5)
    assert (count["wins"], count["ties"], count["losses"]) == (5, 5, 10)
    assert loss["p_unadjusted"] == count["p_unadjusted"]


def test_all_ties_and_missing_seeds():
    assert contrast([1]*20, [1]*20, higher=False)["p_unadjusted"] == 1.0
    with pytest.raises(ValueError):
        contrast([1]*19, [1]*20, higher=True)


def test_legacy_audit_does_not_mask_failures():
    assert returned_batches({"request_epochs": [0, 20], "llm_failures": 0}) == 2
    with pytest.raises(RuntimeError):
        returned_batches({"request_epochs": [0, 20], "llm_failures": 1})
    assert returned_batches({"successful_request_epochs": [0], "llm_failures": 1}) == 1


def test_generated_control_contrast_matches_frozen_primary():
    directional = json.loads((GENERATED / "flash_logic_directional.json").read_text())
    primary = json.loads((GENERATED / "submission_results.json").read_text())
    old = next(r for r in primary["communication_portfolio"] if r["task"] == "logic")
    new = next(r["yield"] for r in directional["contrasts"]
               if (r["left"], r["right"]) == ("full", "none"))
    assert {k: new[k] for k in ("wins", "ties", "losses")} == {
        k: old[k] for k in ("wins", "ties", "losses")}
    assert new["p_unadjusted"] == pytest.approx(old["wilcoxon_p"], abs=1e-12)
    assert len(directional["database_audit"]) == 80
    assert all("parity" not in r["database"] for r in directional["database_audit"])
