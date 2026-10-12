"""Local metrics/figures and optional existing Comet Experiment logging."""

import csv
from pathlib import Path

from .store import canonical


class Telemetry:
    def __init__(self, root, logger=None):
        self.root, self.logger = Path(root), logger

    def metrics(self, epoch, metrics):
        row = {"epoch": epoch, **metrics}
        with (self.root / "metrics.jsonl").open("a") as stream:
            stream.write(canonical(row) + "\n")
        if self.logger is not None:
            self.logger.log_metrics(metrics, step=epoch)

    def plot(self):
        import json
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib import pyplot as plt
        path = self.root / "metrics.jsonl"
        if not path.exists():
            return
        # A crash can leave an uncommitted epoch's metrics; display its latest committed record.
        rows = {r["epoch"]: r for line in path.read_text().splitlines() if (r := json.loads(line))}
        data = [rows[k] for k in sorted(rows)]
        epochs = [r["epoch"] for r in data]
        fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
        panels = [("Archive/best_loss", "Best canonical loss", "#222222"),
                  ("Archive/qualified_regions", "Successful descriptor regions", "#147d64"),
                  ("Archive/qualified_topologies", "Qualifying archived topologies", "#804b99"),
                  ("Compute/expensive_evaluations", getattr(self, "evaluation_label", "Canonical evaluations"), "#555555")]
        for ax, (key, label, color) in zip(axes.flat, panels):
            ax.plot(epochs, [r.get(key, float("nan")) for r in data], color=color, linewidth=2)
            ax.set(xlabel="Epoch", ylabel=label)
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(alpha=.2)
        fig.savefig(self.root / "progress.png", dpi=180)
        fig.savefig(self.root / "progress.pdf")
        if self.logger is not None:
            self.logger.log_figure(figure_name="Exploration progress", figure=fig)
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(8, 3.5), constrained_layout=True)
        for source, color in (("rl", "#2878b5"), ("llm", "#dc7d22"), ("ga", "#14866d"), ("random", "#777777")):
            ax.plot(epochs, [r.get("Source/" + source + "/gain", 0) for r in data], label=source.upper(), color=color)
        ax.set(xlabel="Epoch", ylabel="Archive gain credited at insertion")
        ax.legend(frameon=False, ncol=4)
        fig.savefig(self.root / "sources.png", dpi=180)
        if self.logger is not None:
            self.logger.log_figure(figure_name="Source archive contributions", figure=fig)
        plt.close(fig)
