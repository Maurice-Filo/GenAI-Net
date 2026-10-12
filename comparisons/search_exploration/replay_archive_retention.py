"""Replay a completed discovery stream through both archive rules, without simulation."""

import argparse
import json
from pathlib import Path
import sqlite3
from types import SimpleNamespace

from RL4CRN.search_exploration.archive import DiscoveryArchive
from RL4CRN.search_exploration.config import ExplorationConfig
from RL4CRN.search_streams.archive import StreamDiscoveryArchive


def replay(path, output):
    path, output = Path(path), Path(output)
    manifest = json.loads((path / "manifest.json").read_text())
    status = json.loads((path / "status.json").read_text())
    if status["status"] != "complete":
        raise ValueError("Retention replay requires a completed discovery stream")
    cfg = ExplorationConfig(**manifest["config"])
    with sqlite3.connect("file:" + str(path / "exploration.sqlite") + "?mode=ro", uri=True) as db:
        designs = {key: json.loads(payload) for key, payload in db.execute("SELECT key,payload FROM designs")}
        events = [json.loads(payload) for (payload,) in db.execute("SELECT payload FROM events WHERE kind='archive_insertion' ORDER BY id")]
    archives = [DiscoveryArchive(cfg), StreamDiscoveryArchive(cfg)]
    stores = []
    for _ in archives:
        counts = {}
        def count(key, n=0, values=counts):
            values[key] = values.get(key, 0) + n
            return values[key]
        stores.append(SimpleNamespace(count=count, save_design=lambda record: None))
    history = []
    for index, event in enumerate(events):
        record = designs[event["candidate_hash"]]
        for archive, store in zip(archives, stores):
            archive.add(record, store)
        if index + 1 == len(events) or events[index + 1]["epoch"] != event["epoch"]:
            history.append({"epoch": event["epoch"], "records": index + 1,
                "original_retained": sum(r["qualified"] for r in archives[0].entries.values()),
                "corrected_retained": sum(r["qualified"] for r in archives[1].entries.values())})
    import torch
    checkpoint = torch.load(path / "checkpoint.pt", map_location="cpu", weights_only=False)
    original = checkpoint["archive"]["entries"]
    exact = (set(original) == set(archives[0].entries) and all(
        original[k]["candidate_hash"] == archives[0].entries[k]["candidate_hash"] for k in original))
    if not exact:
        raise AssertionError("Original archive replay does not reproduce the stored terminal membership")
    summary = {"reference": str(path), "original_membership_exact": exact,
        "discovered_qualifying_topologies": len({r["topology_hash"] for r in designs.values() if r["qualified"]}),
        "original_retained": history[-1]["original_retained"],
        "corrected_retained": history[-1]["corrected_retained"], "history": history,
        "interpretation": "Fixed observed discovery sequence: isolates retention, not the changed feedback/learning trajectory of a rerun. No model or evaluator calls."}
    output.mkdir(parents=True, exist_ok=False)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.ticker import MaxNLocator
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    for field, label, color in (("original_retained", "Original retention", "#d07127"),
                                ("corrected_retained", "Corrected retention", "#2673b8")):
        ax.step([r["epoch"] for r in history], [r[field] for r in history], where="post", label=label, color=color)
    ax.set(xlabel="Recorded epoch (zero-based)", ylabel="Retained qualifying topologies")
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    ax.grid(alpha=.2)
    fig.savefig(output / "retention.png", dpi=180)
    fig.savefig(output / "retention.pdf")
    plt.close(fig)
    print(json.dumps({k: v for k, v in summary.items() if k != "history"}, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    replay(args.run, args.output)
