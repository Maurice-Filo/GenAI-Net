"""Compare completed GA runs with the RL run supplying their identical proposals."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from comparisons.llm_seeded_evolution.campaign import DEFAULT_ROOT, write_json
from comparisons.llm_seeded_evolution.replay import sha256
from paper.iclr2027_genai_net_llm.analyze_contract_v2_primary import holm, wilcoxon_p


def collect(plan):
    rows = []
    for job in plan["jobs"]:
        output = Path(job["output"])
        if not (output / "completed.json").exists():
            continue
        result = json.loads((output / "completed.json").read_text())
        replay = json.loads((output / "replay.json").read_text())
        config = json.loads((output / "config.json").read_text())
        for name, field in (("replay.json", "replay_sha256"), ("config.json", "config_sha256"),
                            ("terminal_archive.json", "terminal_archive_sha256")):
            if sha256(output / name) != result[field]:
                raise ValueError(f"Changed completed artifact: {output / name}")
        if result["replay_sha256"] != job["replay_sha256"]:
            raise ValueError("GA did not use the planned proposal stream")
        if (result["pilot"] or result["status"] != "complete"
                or result["search_evaluations"] != replay["search_budget"]
                or result["llm_evaluations"] != replay["llm_evaluations"]
                or result["candidate_evaluations"] != replay["search_budget"] + replay["llm_evaluations"]
                or config["evolution"]["batch_size"] != 1023
                or replay["communication_mode"] != "llm-to-rl"):
            raise ValueError("Budget, batch size or proposal source mismatch")
        archive = json.loads((output / "terminal_archive.json").read_text())
        threshold = result["quality_threshold"]
        def endpoints(records):
            best = {}
            for record in records:
                key = record["topology_hash"]
                best[key] = min(best.get(key, float("inf")), record["loss"])
            losses = sorted(best.values())[:result["portfolio_top_k"]]
            return min(losses), sum(loss < threshold for loss in losses)
        ga_loss, ga_yield = endpoints(archive)
        rl_loss, rl_yield = endpoints(replay["source_terminal_archive"])
        if ga_loss != result["best_loss"] or ga_yield != result["qualified_topologies"]:
            raise ValueError("Terminal archive disagrees with completion summary")
        rows.append({"cohort": job["cohort"], "seed": job["seed"], "ga_loss": ga_loss,
            "rl_loss": rl_loss, "ga_qualified": ga_yield, "rl_qualified": rl_yield,
            "search_evaluations": result["search_evaluations"],
            "replayed_proposals": result["llm_evaluations"], "replay_sha256": result["replay_sha256"]})
    return rows


def report(root):
    plan = json.loads((root / "plan.json").read_text())
    rows = collect(plan)
    summary = {"completed": len(rows), "planned": len(plan["jobs"]),
        "comparison": plan["comparison"], "exploratory": True,
        "inference": "Four Holm-adjusted endpoint tests only when both 20-seed cohorts complete",
        "cohorts": {}, "tests": []}
    rng = np.random.default_rng(20260909)
    for cohort in ("local_logic", "flash_logic"):
        members = sorted([r for r in rows if r["cohort"] == cohort], key=lambda r: r["seed"])
        summary["cohorts"][cohort] = {"n": len(members)}
        if members:
            summary["cohorts"][cohort].update({k: float(np.mean([r[k] for r in members]))
                for k in ("ga_loss", "rl_loss", "ga_qualified", "rl_qualified")})
        if len(rows) != 40:
            continue
        for metric in ("loss", "qualified"):
            ga = np.array([r[f"ga_{metric}"] for r in members])
            rl = np.array([r[f"rl_{metric}"] for r in members])
            differences = ga - rl
            ties = np.isclose(ga, rl, rtol=1e-9, atol=1e-12) if metric == "loss" else ga == rl
            improvement = (rl - ga) if metric == "loss" else differences
            boot = rng.choice(differences, size=(20000, len(members)), replace=True).mean(axis=1)
            summary["tests"].append({"cohort": cohort, "metric": metric, "n": len(members),
                "mean_ga_minus_rl": float(differences.mean()),
                "ci95": np.quantile(boot, [0.025, 0.975]).tolist(),
                "ga_wins_ties_losses": [int(((improvement > 0) & ~ties).sum()),
                    int(ties.sum()), int(((improvement < 0) & ~ties).sum())],
                "p_raw": wilcoxon_p(ga.tolist(), rl.tolist())})
    for row, adjusted in zip(summary["tests"], holm([r["p_raw"] for r in summary["tests"]])):
        row["p_holm_4"] = adjusted
    write_json(root / "comparison.json", summary)
    if rows:
        with (root / "endpoints.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    print(json.dumps(report(parser.parse_args().root), indent=2))
