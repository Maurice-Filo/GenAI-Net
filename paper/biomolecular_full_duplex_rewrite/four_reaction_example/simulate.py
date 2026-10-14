"""Replay a selected four-reaction Logic design from frozen inputs.

Run: .venv/bin/python paper/biomolecular_full_duplex_rewrite/four_reaction_example/simulate.py
No optimization or model calls. Uses the manuscript's CVODE evaluator.
"""
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from comparisons.search_exploration.run_logic5_portfolio_diagnostics import (
    initialize, action_key, replay_with_orders,
)
from comparisons.search_exploration.run_rpa_perturbations import perturb
from paper.iclr2027_genai_net_llm.analyze_logic_correctness_pruning import logic_accuracy

HERE = Path(__file__).resolve().parent


def write_csv(name, rows):
    with (HERE / name).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    design = json.loads((HERE / "design.json").read_text())
    candidate, saved = design["candidate"], design["saved_core"]
    initialize(design)
    ids = [candidate["reaction_ids"][i] for i in saved["retained_indices"]]
    rates = [candidate["rates"][i] for i in saved["retained_indices"]]
    original, original_attempts, ok = replay_with_orders(
        action_key(candidate["reaction_ids"], candidate["rates"]), candidate["saved_orders"])
    assert ok and np.isclose(original["loss"], candidate["loss"], rtol=1e-7, atol=1e-7)
    core, attempts, ok = replay_with_orders(action_key(ids, rates), candidate["saved_orders"])
    assert ok and core["topology_hash"] == saved["topology_hash"]
    assert np.isclose(core["loss"], saved["loss"], rtol=1e-7, atol=1e-7)
    accuracy = logic_accuracy(core["outputs"], core["targets"])
    deviation = float(np.abs(core["outputs"] - original["outputs"]).max())
    assert accuracy["all16_correct"] and deviation <= .01 and core["loss"] < design["threshold"]
    t = np.linspace(0, 100, 1000)
    outputs = core["outputs"][:, 0, :]
    original_outputs = original["outputs"][:, 0, :]
    targets = core["targets"][:, 0].astype(int)
    labels = ["".join(str(int(v)) for v in row) for row in core["inputs"]]
    perturbation_rows, perturbed_outputs = [], []
    for draw, direction in enumerate(design["directions"]):
        varied = perturb(ids, rates, 1.1, direction)
        value, draw_attempts, ok = replay_with_orders(action_key(ids, varied), candidate["saved_orders"])
        correct = ok and logic_accuracy(value["outputs"], value["targets"])["all16_correct"]
        valid = bool(correct and value["loss"] < design["threshold"])
        perturbation_rows.append({"draw": draw, "status": value["status"],
                                  "loss": value["loss"], "all16_correct": bool(correct),
                                  "valid": valid, **{f"k_{rid}": rate for rid, rate in zip(ids, varied)}})
        perturbed_outputs.append(value["outputs"][:, 0, :])
    varied_outputs = np.asarray(perturbed_outputs)
    passes = sum(row["valid"] for row in perturbation_rows)
    assert passes / 32 == design["membership"]["core_pass_fraction"]
    write_csv("perturbations.csv", perturbation_rows)
    write_csv("truth_table.csv", [{"inputs": label, "target": int(target),
              "output_t100": float(y[-1]), "original_output_t100": float(z[-1]),
              "correct": bool((2*target-1)*(y[-1]-.5)>1e-6)}
              for label, target, y, z in zip(labels, targets, outputs, original_outputs)])
    write_csv("trajectories.csv", [{"inputs": labels[i], "target": int(targets[i]),
              "time": float(time), "output_four_reactions": float(outputs[i,j]),
              "output_five_reactions": float(original_outputs[i,j])}
              for i in range(16) for j, time in enumerate(t)])
    np.savez_compressed(HERE / "simulation.npz", time=t, inputs=core["inputs"],
                        initial_conditions=core["initial_conditions"], targets=targets,
                        outputs=outputs, original_outputs=original_outputs,
                        perturbed_outputs=varied_outputs)
    result = {"fresh_simulation": True, "solver": "CVODE", "rtol": 1e-8, "atol": 1e-8,
              "initial_conditions": core["initial_conditions"].tolist(),
              "reaction_ids": ids, "rates": rates, "loss": core["loss"],
              "original_loss": original["loss"], "qualification_threshold": design["threshold"],
              "max_output_deviation_from_original": deviation, **accuracy,
              "perturbation_passes": passes, "perturbation_tests": 32,
              "perturbation_protocol": "Replay the saved 16 directions and their negatives; k'=k*exp(z*log(1.1)); no clipping or refitting.",
              "core_replay_attempts": attempts, "original_replay_attempts": original_attempts}
    (HERE / "results.json").write_text(json.dumps(result, indent=2) + "\n")

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    for target, color in [(0, "#b55c24"), (1, "#187b85")]:
        rows = outputs[targets == target]
        for row in rows:
            axes[0].plot(t, row, color=color, alpha=.6, linewidth=1.3)
    axes[0].axhline(.5, color="#555555", ls="--", lw=1)
    axes[0].set(xlabel="Time (model units)", ylabel="OUT concentration (model units)",
                title="Output trajectories for all 16 inputs", xlim=(0, 100))
    axes[0].legend(handles=[Line2D([], [], color="#187b85", label="Target 1 (8 inputs)"),
                           Line2D([], [], color="#b55c24", label="Target 0 (8 inputs)"),
                           Line2D([], [], color="#555555", ls="--", label="Classification threshold")],
                   loc="lower right", fontsize=8)
    colors = ["#187b85" if target else "#b55c24" for target in targets]
    axes[1].bar(np.arange(16), outputs[:, -1], color=colors)
    axes[1].scatter(np.arange(16), targets, marker="_", s=100, color="black", label="Target", zorder=3)
    axes[1].axhline(.5, color="#555555", ls="--", lw=1)
    axes[1].set(xticks=np.arange(16), xticklabels=labels, xlabel="Inputs (u1 u2 u3 u4)",
                ylabel="OUT at t = 100", title="All 16 Boolean outputs are correct")
    axes[1].tick_params(axis="x", rotation=90)
    axes[1].legend(fontsize=8)
    fig.suptitle(f"Selected four-reaction Logic design | full duplex, seed 13 | loss {core['loss']:.5f}")
    for suffix in ("png", "pdf"):
        fig.savefig(HERE / f"overview.{suffix}", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(4, 4, figsize=(12, 10), sharex=True, sharey=True, layout="constrained")
    for i, ax in enumerate(axes.flat):
        color = colors[i]
        ax.fill_between(t, varied_outputs[:, i].min(axis=0), varied_outputs[:, i].max(axis=0),
                        color=color, alpha=.18)
        ax.plot(t, original_outputs[i], color="#777777", ls=":", lw=2)
        ax.plot(t, outputs[i], color=color, lw=1.3)
        ax.axhline(.5, color="#555555", ls="--", lw=.8)
        ax.set_title(f"{labels[i]} → {targets[i]}   |   final {outputs[i,-1]:.4f}", fontsize=10)
        ax.set(xlim=(0, 100), ylim=(0, max(1.2, float(varied_outputs.max())*1.03)))
    fig.supxlabel("Time (model units)")
    fig.supylabel("OUT concentration (model units)")
    fig.suptitle("Four-reaction design: every input combination\n"
                 f"Solid: four reactions · Dotted: original five · Shading: range of 32 rate perturbations ({passes}/32 pass)",
                 fontsize=12)
    for suffix in ("png", "pdf"):
        fig.savefig(HERE / f"all_inputs.{suffix}", dpi=180)
    plt.close(fig)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
