"""Read-only provenance of the rate vectors supporting robust canonical Logic cores."""

import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

from comparisons.search_exploration.analyze_logic5_four_arm import ARMS, write_csv
from comparisons.search_exploration.analyze_rpa_llm_provenance import connect
from comparisons.search_exploration.plot_logic5_canonical_topology_counts import canonical_hash
from RL4CRN.search_exploration.store import canonical

GROUPS = ("llm_direct", "llm_fit", "rl_direct", "rl_fit", "ga_rl", "ga_llm", "other", "multiple", "unknown")
LABELS = ("Direct LLM", "LLM + AdaGrad", "RL-sampled", "RL + AdaGrad", "Mutation: RL lineage",
          "Mutation: LLM lineage", "Other search", "Multiple observed paths", "Unresolved provenance")
COLORS = ("#a64e85", "#d08dae", "#168577", "#68b3a1", "#376ab3", "#bc7838", "#898989", "#56516e", "#ce3c49")


def parameter_hash(candidate, selection):
    records = [{k: r[k] for k in ("reaction_id", "structure", "parameters")} for r in selection["fixed"]]
    records += [{**selection["catalog"][str(i)], "parameters": [k]}
                for i, k in zip(candidate["reaction_ids"], candidate["rates"])]
    return hashlib.sha256(canonical(sorted(records, key=canonical)).encode()).hexdigest()


def route(record):
    fitted = record.get("stream") == "optimization"
    source, origin = record.get("source"), record.get("origin")
    if source == "llm":
        return "llm_fit" if fitted else "llm_direct"
    if source == "rl":
        return "rl_fit" if fitted else "rl_direct"
    if source == "ga":
        return {"rl": "ga_rl", "llm": "ga_llm"}.get(origin, "other")
    return "other" if source == "random" else "unknown"


def merged_group(paths):
    paths = set(paths)
    return next(iter(paths)) if len(paths) == 1 else "multiple" if paths else "unknown"


def extract(root):
    summary_path = root / "analysis/summary.json"
    summary = json.loads(summary_path.read_text())
    if summary["status"] != "complete" or summary["baseline_unresolved"]:
        raise ValueError("Require completed resolved diagnostics")
    with gzip.open(root / "selection.json.gz", "rt") as stream:
        selection = json.load(stream)
    plan = json.loads((root / "plan.json").read_text())
    if hashlib.sha256((root / "plan.json").read_bytes()).hexdigest() != summary["plan_sha256"]:
        raise ValueError("Diagnostic plan mismatch")
    if hashlib.sha256((root / "selection.json.gz").read_bytes()).hexdigest() != plan["input_hashes"][str(root / "selection.json.gz")]:
        raise ValueError("Frozen selection changed")
    selected = defaultdict(list)
    for member in summary["memberships"]:
        if member["core_pass_fraction"] >= plan["robust_diagnostic_cutoff"]:
            selected[member["arm"], member["seed"]].append(member)
    witnesses, core_rows, parents, inputs = [], [], [], []
    for n, run in enumerate(selection["runs"], 1):
        arm, seed, folder = run["arm"], run["seed"], Path(run["output"])
        members = selected[arm, seed]
        candidates = {m["candidate"]: selection["candidates"][m["candidate"]] for m in members}
        keys = {identifier: parameter_hash(c, selection) for identifier, c in candidates.items()}
        key_set = set(keys.values())
        direct = defaultdict(list)
        bank = folder / "direct_llm_proposals.jsonl"
        if bank.exists():
            for line in bank.read_text().splitlines():
                entry = json.loads(line)
                record = entry.get("record")
                if record and record["candidate_hash"] in key_set:
                    if not entry["direct_supplied_rate_only"] or not record["simulation_complete"]:
                        raise ValueError("Unexpected direct proposal record")
                    direct[record["candidate_hash"]].append({"route": "llm_direct", "source": "llm", "origin": "llm",
                        "kind": "direct_bank", "completed_epoch": entry["completed_epoch"], "loss": record["loss"]})
        events = defaultdict(list)
        designs = {}
        with connect(folder) as db:
            if key_set:
                placeholders = ",".join("?" for _ in key_set)
                for key, payload in db.execute(f"SELECT key,payload FROM designs WHERE key IN ({placeholders})", tuple(key_set)):
                    designs[key] = json.loads(payload)
                for eid, payload in db.execute(f"SELECT id,payload FROM events WHERE kind='archive_insertion' AND json_extract(payload,'$.candidate_hash') IN ({placeholders}) ORDER BY id", tuple(key_set)):
                    event = json.loads(payload)
                    event.update(event_id=eid, kind="search_event", route=route(event))
                    events[event["candidate_hash"]].append(event)
            for identifier, key in keys.items():
                candidate = candidates[identifier]
                if key in designs:
                    saved = designs[key]
                    if (parameter_hash(saved, selection) != key or saved["topology_hash"] != candidate["topology_hash"] or
                            not np.isclose(saved["loss"], candidate["loss"], rtol=1e-7, atol=1e-7)):
                        raise ValueError("Exact selected design does not match frozen diagnostics")
                if not events[key] and not direct[key]:
                    raise ValueError(f"No provenance evidence for {folder}/{identifier}")
                for event in events[key]:
                    if not np.isclose(event["loss"], candidate["loss"], rtol=1e-7, atol=1e-7):
                        raise ValueError("Candidate event score differs from selected vector")
                    parent_hash = event.get("parent_hash")
                    if parent_hash:
                        row = db.execute("SELECT payload FROM designs WHERE key=?", (parent_hash,)).fetchone()
                        parent = json.loads(row[0]) if row else None
                        parents.append({"arm": arm, "seed": seed, "candidate": identifier, "event_id": event["event_id"],
                            "route": event["route"], "parent_hash": parent_hash, "parent_available": parent is not None,
                            "same_candidate_as_parent": parent_hash == key,
                            "topology_changed": None if parent is None else candidate["topology_hash"] != parent["topology_hash"],
                            "loss_improved": None if parent is None else event["loss"] < parent["loss"],
                            "parent_loss": None if parent is None else parent["loss"], "child_loss": event["loss"]})
        cores = defaultdict(list)
        for member in members:
            identifier = member["candidate"]
            candidate = candidates[identifier]
            key = keys[identifier]
            evidence = events[key] + direct[key]
            paths = sorted({e["route"] for e in evidence})
            record_path = root / "records" / f"{identifier}.json"
            record = json.loads(record_path.read_text())
            if record["plan_sha256"] != summary["plan_sha256"] or record["status"] != "complete":
                raise ValueError("Unverified robustness record")
            core = record["cores"]["0.01"]
            if core["topology_hash"] != member["core_hash"]:
                raise ValueError("Core membership mismatch")
            core_hash = canonical_hash(json.loads(core["structure_json"]))
            witness = {"arm": arm, "seed": seed, "candidate": identifier, "candidate_hash": key,
                "topology_hash": candidate["topology_hash"], "core_hash": core_hash,
                "core_pass_fraction": member["core_pass_fraction"], "retained_reactions": member["retained_reactions"],
                "loss": candidate["loss"], "routes": paths, "group": merged_group(paths),
                "source": designs.get(key, {}).get("source"), "origin": designs.get(key, {}).get("origin"),
                "parameter_operator": designs.get(key, {}).get("parameter_operator"),
                "search_event_count": len(events[key]), "direct_bank_count": len(direct[key]),
                "first_search_event": events[key][0] if events[key] else None,
                "evidence": evidence, "folder": str(folder)}
            witnesses.append(witness)
            cores[core_hash].append(witness)
        expected = next(r["robust_cores"] for r in summary["rows"] if r["arm"] == arm and r["seed"] == seed)
        if len(cores) != expected:
            raise ValueError("Provenance cores do not reproduce audited robust counts")
        for core_hash, bank_witnesses in cores.items():
            paths = sorted({r for w in bank_witnesses for r in w["routes"]})
            core_rows.append({"arm": arm, "seed": seed, "core_hash": core_hash, "group": merged_group(paths),
                              "routes": paths, "witnesses": [w["candidate"] for w in bank_witnesses]})
        inputs.append({"folder": str(folder), "database_size": (folder / "exploration.sqlite").stat().st_size,
                       "database_mtime_ns": (folder / "exploration.sqlite").stat().st_mtime_ns,
                       "direct_bank_sha256": hashlib.sha256(bank.read_bytes()).hexdigest() if bank.exists() else None})
        print(f"Audited {n}/80 runs: {arm}/{seed}, {len(cores)} robust cores", flush=True)
    counts = [{"arm": arm, "seed": seed, **{g: sum(r["group"] == g for r in core_rows if r["arm"] == arm and r["seed"] == seed)
                                            for g in GROUPS}} for arm in ARMS for seed in range(1, 21)]
    pooled = []
    for arm in ARMS:
        bank = defaultdict(set)
        for core in core_rows:
            if core["arm"] == arm:
                bank[core["core_hash"]].update(core["routes"])
        counter = Counter(merged_group(v) for v in bank.values())
        pooled.append({"arm": arm, "total": len(bank), **{g: counter[g] for g in GROUPS}})
    return {"status": "complete", "counts": counts, "pooled": pooled, "cores": core_rows, "witnesses": witnesses,
            "parents": parents, "inputs": inputs, "source": str(summary_path),
            "source_sha256": hashlib.sha256(summary_path.read_bytes()).hexdigest(),
            "plan_sha256": summary["plan_sha256"], "new_evaluations": 0}


def render(output, result, *, title="Recorded paths to parameter-robust Logic-5 cores",
           include_footer=True, include_subtitle=True, stem="LOGIC5_ROBUST_PROVENANCE",
           structure_name="cores"):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from comparisons.search_exploration.plot_logic5_portfolio_robustness import LABELS as ARMLABELS
    plt.rcParams.update({"font.size": 10, "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.5))
    fig.subplots_adjust(left=.07, right=.96, bottom=.29 if include_footer else .22, top=.78, wspace=.26)
    fig.text(.07, .94, title, fontsize=17, weight="bold")
    if include_subtitle:
        fig.text(.07, .88, "Attribution follows evaluated rate vectors; shared paths are not assigned an arbitrary single origin.", fontsize=10)
    means = {arm: {g: np.mean([r[g] for r in result["counts"] if r["arm"] == arm]) for g in GROUPS} for arm in ARMS}
    for ax, pooled in zip(axes, (False, True)):
        bottom = np.zeros(4)
        for group, label, color in zip(GROUPS, LABELS, COLORS):
            values = np.array([next(r[group] for r in result["pooled"] if r["arm"] == arm) if pooled else means[arm][group] for arm in ARMS])
            if not values.any():
                continue
            ax.bar(range(4), values, bottom=bottom, color=color, width=.65, label=label)
            bottom += values
        for i, total in enumerate(bottom):
            ax.text(i, total + max(bottom)*.025, f"{total:.0f}" if pooled else f"{total:.2f}", ha="center", weight="bold")
        ax.set_xticks(range(4), [ARMLABELS[a] for a in ARMS])
        ax.set_ylim(0, max(bottom)*1.16)
        ax.set_ylabel(f"Unique robust {structure_name} across 20 seeds" if pooled
                      else f"Mean unique robust {structure_name} per seed")
        ax.set_title("B  Pooled union across seeds" if pooled else "A  Per-seed portfolios", loc="left", weight="bold", pad=15)
        ax.grid(axis="y", alpha=.15); ax.set_axisbelow(True)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.5,.095 if include_footer else .035), ncol=4, frameon=False, fontsize=9)
    if include_footer:
        fig.text(.07, .055, "Robust = >=29/32 passing 10% perturbations after pruning. Reaction order and rates do not create additional topology counts.", fontsize=9)
        fig.text(.07, .025, "RL ancestry through imitation is not explicitly traceable. Mutation labels use recorded parent lineage; all attributions are descriptive.", fontsize=9)
    for ext in ("pdf", "png"):
        fig.savefig(output / f"{stem}.{ext}", dpi=220)
    plt.close(fig)


def report(output, result):
    lines = ["# Robust Logic-5 Core Provenance", "", "All 80 completed runs; no new simulations, fitting or LLM calls.", "",
        "## Mean Unique Robust Cores Per Seed", "",
        "| Arm | " + " | ".join(LABELS) + " | Total |",
        "|---|" + "---:|" * (len(GROUPS)+1)]
    for arm in ARMS:
        means = [np.mean([r[g] for r in result["counts"] if r["arm"] == arm]) for g in GROUPS]
        lines.append(f"| {arm} | " + " | ".join(f"{x:.2f}" for x in means) + f" | {sum(means):.2f} |")
    full = {g: np.mean([r[g] for r in result["counts"] if r["arm"] == "full"]) for g in GROUPS}
    lines += ["", "## Main Descriptive Finding", "",
        f"In full duplex, {full['llm_fit']:.2f} robust cores per seed are attributed exclusively to LLM proposals followed by AdaGrad, {full['rl_fit']:.2f} to RL-sampled structures followed by AdaGrad, and {full['llm_direct']:.2f} to direct supplied-rate LLM proposals. Explicit mutation paths with RL and LLM ancestry contribute {full['ga_rl']:.2f} and {full['ga_llm']:.2f} respectively; {full['multiple']:.2f} have multiple observed paths.",
        "The robust-core increase is descriptively concentrated in LLM-seeded rate refinement. This is not evidence that AdaGrad was necessary for the parent's robustness: parent losses were checked, but parent robustness was not counterfactually evaluated. Similarly, this decomposition does not isolate a causal benefit of elite imitation or quantify unlogged RL-to-LLM influence.",
        "No exclusively RL-sampled, unrefined witness supports a robust core in this selected best-vector analysis. This must not be generalized to all sampled RL networks or all their parameterizations, which were not subjected to the same whole-portfolio perturbation study."]
    lines += ["", "## Attribution Rules and Limits", "",
        "We trace the exact original rate vectors whose post-hoc cores passed robustness testing. Pruning itself is a diagnostic, not an RL or LLM generation event. Each canonical core is counted once per seed; route sets are merged across all robust witnesses. Pooled counts also merge across seeds. Multi-route cores have a separate category, not duplicated credit.",
        "Direct LLM means the exact supplied-rate proposal is present in the direct bank or an unfitted LLM insertion event. LLM + AdaGrad and RL + AdaGrad require an explicit optimization-stream event. RL-sampled denotes direct policy sampling, not proof of independence from prior LLM training signals. GA mutation preserves recorded initial lineage (RL or LLM); these paths may also be rate-fitted, recorded separately in event evidence.",
        "Design rows can be overwritten on rediscovery, so current source/origin labels are retained only as auxiliary metadata. Classification uses all matching search journal events and direct-bank evidence for the exact reaction/rate vector. Multiple paths do not mean multiple independent causal contributions. Event order within the search journal is saved, but no unique global first inventor is inferred across asynchronous search and LLM logs.",
        "The origin label of an RL sample resets to RL even if elite imitation previously learned from LLM proposals. Thus absence of an explicit LLM parent does not demonstrate absence of LLM influence. Conversely an LLM can use HOF context without an explicit structural parent. This analysis cannot identify those mediated causal paths.",
        "Parent checks report structural change and loss improvement only when the saved parent is available. Parent rows do not establish that the parent was itself robust; unselected parameter vectors were not retested. An identical parent/child candidate can occur if fitting makes no change and is explicitly flagged, not claimed as improvement.",
        "Robustness and identity follow the completed canonical/recorded-order fallback protocol: finite 32-draw diagnostic, 10% multiplicative perturbations, no refitting, output-preserving pruning at epsilon 0.01. A topology is counted if at least one observed witness passes, not if all rate vectors pass. No new significance tests are introduced.",
        f"Witness vectors: {len(result['witnesses'])}; unique seed/core entries: {len(result['cores'])}.",
        f"Source: `{result['source']}`", f"Source SHA256: `{result['source_sha256']}`"]
    (output / "REPORT.md").write_text("\n".join(lines)+"\n")
    write_csv(output / "core_attribution.csv", [{**r, "routes": ";".join(r["routes"]),
        "witnesses": ";".join(r["witnesses"])} for r in result["cores"]])
    write_csv(output / "witness_attribution.csv", [{k: v for k, v in w.items()
        if k not in ("evidence", "first_search_event")} | {"routes": ";".join(w["routes"])} for w in result["witnesses"]])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = extract(args.root.resolve())
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    write_csv(args.output / "counts_per_seed.csv", result["counts"])
    write_csv(args.output / "pooled_counts.csv", result["pooled"])
    if result["parents"]:
        write_csv(args.output / "parent_checks.csv", result["parents"])
    render(args.output, result)
    report(args.output, result)
