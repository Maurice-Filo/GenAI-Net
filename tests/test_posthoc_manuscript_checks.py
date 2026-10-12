import copy
import csv
import json

import numpy as np
import pytest

from paper.iclr2027_genai_net_llm import generate_posthoc_checks as checks
from paper.iclr2027_genai_net_llm.analyze_contract_v2_primary import holm, wilcoxon_p


@pytest.fixture(scope="module")
def payload():
    return json.loads(checks.SOURCE.read_text())


def test_fingerprints_and_complete_statistical_families(payload):
    checks.verify(payload)
    for family, field, size in checks.FAMILIES:
        rows = checks.family_rows(payload, family)
        assert len(rows) == size
        np.testing.assert_allclose([r[field] for r in rows], holm([r["p_raw"] for r in rows]))


def test_export_rejects_incorrect_correction(payload):
    bad = copy.deepcopy(payload)
    bad["overlap_contrasts"][0]["p_holm_4"] = -1
    with pytest.raises(ValueError, match="Incorrect Holm"):
        checks.verify(bad)


def test_export_rejects_stale_inputs(payload):
    bad = copy.deepcopy(payload)
    bad["input_sha256"]["analysis_plan_v2.json"] = "stale"
    with pytest.raises(ValueError, match="Stale post-hoc input"):
        checks.verify(bad)


def test_endpoint_statistics_match_existing_analysis_helper(payload):
    for cohort, name, count, loss in (
        ("flash_logic", "flash_logic_directional_endpoints.csv", "qualified", "best_loss"),
        ("local_logic", "local_half_duplex_endpoints.csv", "qualified_topologies", "rank_one_loss"),
    ):
        with (checks.GENERATED / name).open() as stream:
            idx = {(r["arm"], int(r["seed"])): r for r in csv.DictReader(stream)
                   if r.get("task", "logic") == "logic"}
        for row in payload["directional"]:
            if row["cohort"] != cohort:
                continue
            metric = loss if row["metric"] in ("best_loss", "rank_one_loss") else count
            left, right = [[float(idx[arm, s][metric]) for s in range(20)]
                           for arm in (row["left"], row["right"])]
            assert row["mean_delta"] == pytest.approx(np.mean(np.subtract(left, right)))
            assert row["p_raw"] == pytest.approx(wilcoxon_p(left, right))


def test_overlap_counts_and_schedules(payload):
    rows = payload["overlap_runs"]
    assert len(rows) == 160
    idx = {(r["cohort"], r["arm"], r["seed"]): r for r in rows}
    for row in rows:
        assert row["unique"] == (row["first_seen_in_exposed_hof"]
                                 + row["first_seen_absent_from_exposed_hof"])
        if row["arm"] in ("none", "llm_to_rl"):
            assert row["first_seen_in_exposed_hof"] == 0
    for cohort in ("flash_logic", "local_logic"):
        for seed in range(20):
            left, right = [idx[cohort, arm, seed] for arm in ("llm_to_rl", "none")]
            assert left["proposal_bank_sha256"] != right["proposal_bank_sha256"]
            assert left["completion_epochs"] != right["completion_epochs"]
    for row in payload["overlap_contrasts"]:
        left, right = [[idx[row["cohort"], arm, s]["first_seen_absent_from_exposed_hof"]
                        for s in range(20)] for arm in (row["left"], row["right"])]
        assert row["p_raw"] == pytest.approx(wilcoxon_p(left, right))


def test_classifier_counts_recomputed_at_seed_level(payload):
    data = json.loads((checks.GENERATED / "classifier_threshold_accuracy.json").read_text())
    threshold = json.loads((checks.GENERATED / "analysis_plan_v2.json").read_text())[
        "quality_thresholds"]["classifier"]
    records = data["records"]
    assert len(records) == 1200
    for row in payload["classifier"]["contrasts"]:
        counts = {arm: [0] * 20 for arm in ("none", "full")}
        for record in records:
            outputs, targets = np.asarray(record["outputs"]), np.asarray(record["targets"])
            if row["metric"] == "binary_correct":
                correct = np.array_equal(outputs >= 0.5, targets)
            else:
                signs = 2 * targets[:, 0] - 1
                correct = bool(np.all((outputs[:, 0] - outputs[:, 1]) * signs > 1e-6))
            if correct and record["loss"] < threshold * row["multiplier"]:
                counts[record["arm"]][record["seed"]] += 1
        assert row["left_mean"] == pytest.approx(np.mean(counts["full"]))
        assert row["right_mean"] == pytest.approx(np.mean(counts["none"]))
        assert row["p_raw"] == pytest.approx(wilcoxon_p(counts["full"], counts["none"]))


def test_generated_tables_match_verified_source(payload):
    for name, content in checks.render(payload).items():
        assert (checks.GENERATED / name).read_text() == content
