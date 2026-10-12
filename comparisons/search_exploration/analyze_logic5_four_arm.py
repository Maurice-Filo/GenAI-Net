"""Paired post-hoc inference from completed Logic-5 four-arm endpoint artifacts."""

import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

from comparisons.search_exploration.analyze_rpa_statistical import paired, holm
from comparisons.search_exploration.recover_rpa_directional_balance import request_audit

ARMS = ("none", "llm-to-rl", "rl-to-llm", "full")
LABELS = {"none": "No communication", "llm-to-rl": "LLM to search",
          "rl-to-llm": "Search to LLM", "full": "Full duplex"}
COLORS = {"none": "#6b6975", "llm-to-rl": "#bc7838", "rl-to-llm": "#376ab3", "full": "#a64e85"}


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def extract(root):
    plan, progress = read(root / "plan.json"), read(root / "progress.json")
    if progress["status"] != "complete" or progress["completed"] != 80:
        raise ValueError("Require the complete 80-run cohort")
    if sha(root / "plan.json") != (root / "plan.sha256").read_text().strip():
        raise ValueError("Frozen plan hash mismatch")
    rows, reference = [], {}
    for job in plan["jobs"]:
        folder = Path(job["output"])
        manifest, completed, union = [read(folder / name) for name in
                                    ("manifest.json", "completed.json", "terminal_union.json")]
        cfg, metrics, counters = manifest["config"], completed["metrics"], completed["counters"]
        if (completed["status"] != "complete" or completed["epochs"] != 100 or
                manifest["task"] != "logic" or manifest["communication_mode"] != job["arm"] or
                cfg["seed"] != job["seed"] or not manifest["withhold_initial_hof"]):
            raise ValueError(f"Endpoint identity mismatch: {folder}")
        if (counters["expensive_evaluations"] > cfg["evaluation_budget"] or
                counters.get("julia/reserved", 0) or metrics.get("Optimization/pending", 0) or
                metrics.get("LLM/pending_rounds", 0)):
            raise ValueError(f"Budget violation or undrained streams: {folder}")
        audit = request_audit(folder)
        if audit["balance_affected"]:
            raise ValueError(f"Balance-contaminated endpoint: {folder}")
        # Compare scientific settings within seed; scheduling and model draws remain unmatched.
        identity = {k: manifest[k] for k in ("config", "task_config", "julia_recipe", "model",
                    "system_prompt_sha256", "task_prompt_sha256", "search_workers", "optimization_workers",
                    "llm_every", "llm_candidates", "llm_timeout", "policy")}
        if job["seed"] in reference and identity != reference[job["seed"]]:
            raise ValueError(f"Within-seed scientific settings mismatch: {folder}")
        reference[job["seed"]] = identity
        loss, count = union["union_best_loss"], union["union_qualified"]
        if not np.isfinite(loss) or loss <= 0 or count < 0 or int(count) != count:
            raise ValueError("Invalid endpoint")
        if not np.isclose(loss, metrics["Union/best_loss"], rtol=1e-12) or count != metrics["Union/qualified_topologies"]:
            raise ValueError("Terminal union disagrees with completion metrics")
        rows.append({"arm": job["arm"], "seed": job["seed"], "loss": loss, "qualified": count,
            "threshold": cfg["quality_threshold"], "evaluations": counters["expensive_evaluations"],
            "requests": audit["requests"], "failed_requests": audit["failed_requests"],
            "retained": job["retained"], "recovery_kind": job["recovery_kind"], "output": str(folder),
            **{name.replace(".json", "") + "_sha256": sha(folder / name) for name in
               ("completed.json", "manifest.json", "terminal_union.json")}})
    if len(rows) != 80 or {(r["arm"], r["seed"]) for r in rows} != set(itertools.product(ARMS, range(1, 21))):
        raise ValueError("Expected all 20 seeds in all four arms, once each")
    if len({r["threshold"] for r in rows}) != 1:
        raise ValueError("Thresholds differ")
    return rows


def values(rows, arm, metric):
    return np.array([r[metric] for r in sorted(rows, key=lambda r: r["seed"]) if r["arm"] == arm])


def exact_pratt_p(difference):
    """Exact sign-permutation distribution via integer rank-sum dynamic programming."""
    d = np.asarray(difference, dtype=float)
    if not np.isfinite(d).all() or len(d) > 20:
        raise ValueError("This exact endpoint test supports at most 20 finite pairs")
    # Pratt ranks include zeros, whose signs have no effect. Average ranks are half-integers.
    ranks = np.rint(2 * rankdata(abs(d))).astype(int)
    nonzero = ranks[d != 0]
    if not len(nonzero):
        return 1.
    distribution = np.zeros(int(nonzero.sum()) + 1, dtype=np.int64)
    distribution[0] = 1
    for rank in nonzero:
        distribution[rank:] += distribution[:-rank].copy()
    positive = int(ranks[d > 0].sum())
    tail = min(positive, int(nonzero.sum()) - positive)
    return min(1., 2 * int(distribution[:tail+1].sum()) / 2**len(nonzero))


def paired_endpoint(a, b, *, smaller, resamples=20000, seed=20260924):
    a, b = np.asarray(a, float), np.asarray(b, float)
    result = paired(a, b, smaller=smaller, inference=False)
    ties = np.isclose(a, b, rtol=1e-9, atol=1e-12) if smaller else a == b
    result["p"] = exact_pratt_p(np.where(ties, 0., a-b))
    indices = np.random.default_rng(seed).integers(0, len(a), size=(resamples, len(a)))
    result["mean_difference_ci95"] = np.quantile((a-b)[indices].mean(axis=1), [.025, .975]).tolist()
    result["median_ratio_ci95"] = (np.quantile(np.median((a/b)[indices], axis=1), [.025, .975]).tolist()
                                      if (b > 0).all() else None)
    return result


def statistics(rows):
    tests = []
    for b, a in itertools.combinations(ARMS, 2):
        for metric in ("loss", "qualified"):
            print(f"Exact paired test: {a} vs {b}, {metric}", flush=True)
            result = paired_endpoint(values(rows, a, metric), values(rows, b, metric), smaller=metric == "loss",
                            resamples=20000, seed=20260924)
            tests.append({"a": a, "b": b, "metric": metric, **result})
    for test, adjusted in zip(tests, holm([t["p"] for t in tests])):
        test["holm_p_12"] = adjusted
    return tests


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot(output, rows, tests):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.ticker import FuncFormatter, FixedLocator, NullFormatter
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 12,
        "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.8))
    x = np.arange(4)
    jitter = np.random.default_rng(20260924).uniform(-.12, .12, 20)
    for ax, metric, title in zip(axes[0], ("loss", "qualified"),
                                ("A  Best endpoint loss", "B  Qualifying topology count")):
        data = np.array([values(rows, arm, metric) for arm in ARMS])
        for s in range(20):
            ax.plot(x+jitter[s], data[:, s], color="#87909a", alpha=.18, lw=.65, zorder=1)
        boxes = ax.boxplot(data.T, positions=x, widths=.43, patch_artist=True, showfliers=False,
                           medianprops={"color": "#24272c", "linewidth": 1.7},
                           whiskerprops={"color": "#87909a"}, capprops={"color": "#87909a"})
        for i, arm in enumerate(ARMS):
            boxes["boxes"][i].set(facecolor=COLORS[arm], alpha=.17, edgecolor=COLORS[arm])
            ax.scatter(i+jitter, data[i], s=23, c=COLORS[arm], alpha=.82, linewidths=0, zorder=3)
        ax.set_xticks(x, ["No\ncommunication", "LLM to\nsearch", "Search to\nLLM", "Full\nduplex"])
        ax.set_title(title, loc="left", weight="bold", pad=13)
        ax.grid(axis="y", alpha=.14)
        if metric == "loss":
            ax.set_yscale("log")
            ax.set_ylabel("Weighted trajectory L1 loss (lower is better)")
            ax.yaxis.set_major_locator(FixedLocator([.004, .006, .01, .02, .03]))
            ax.yaxis.set_major_formatter(FuncFormatter(lambda val, _: f"{val:g}"))
            ax.yaxis.set_minor_formatter(NullFormatter())
        else:
            ax.set_ylim(bottom=0)
            ax.set_ylabel("Distinct qualifying topologies (higher is better)")
    for ax, metric, title in zip(axes[1], ("loss", "qualified"),
                                ("C  Full-duplex loss reduction", "D  Full-duplex topology-count change")):
        selected = [next(t for t in tests if t["a"] == "full" and t["b"] == arm and t["metric"] == metric)
                    for arm in ARMS[:3]]
        intervals, estimates = [], []
        for t in selected:
            if metric == "loss":
                estimates.append(100 * (1-t["median_paired_ratio"]))
                intervals.append([100*(1-t["median_ratio_ci95"][1]), 100*(1-t["median_ratio_ci95"][0])])
            else:
                estimates.append(t["mean_difference"])
                intervals.append(t["mean_difference_ci95"])
        lo, hi = min(0, np.min(intervals)), max(0, np.max(intervals))
        pad = max(hi-lo, 1) * .14
        ax.set_xlim(lo-pad, hi+pad)
        ax.axvline(0, color="#6b6975", ls="--", lw=1)
        for i, (estimate, ci, t) in enumerate(zip(estimates, intervals, selected)):
            y = 2-i
            ax.plot(ci, [y, y], color=COLORS[t["b"]], lw=2.5)
            ax.scatter(estimate, y, c=COLORS["full"], edgecolors="white", s=72, zorder=3)
            ax.text(.99, y+.23, f"Holm p = {t['holm_p_12']:.3g}  |  W/T/L {t['wins']}/{t['ties']}/{t['losses']}",
                    transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=9, color="#414751")
        ax.set_yticks([2, 1, 0], ["vs no communication", "vs LLM to search", "vs search to LLM"])
        ax.set_ylim(-.48, 2.65)
        ax.set_title(title, loc="left", weight="bold", pad=13)
        ax.grid(axis="x", alpha=.14)
        ax.set_xlabel("Median paired loss reduction (%)" if metric == "loss" else "Mean paired difference in topology count")
    fig.suptitle("Logic-5: paired effects of LLM-search communication", x=.035, ha="left", fontsize=17, weight="bold")
    fig.text(.035, .915, "20 matched seeds per arm  |  100 epochs  |  DeepSeek V4 Flash + RL / mutation / AdaGrad", fontsize=11, color="#414751")
    fig.text(.035, .025,
             "Endpoints include the terminal union of search and LLM candidates; qualification threshold = 0.042713.\n"
             "Lines in A/B connect matched seeds. C/D: paired bootstrap 95% intervals (not multiplicity-adjusted).\n"
             "Two-sided exact signed-rank tests; Holm correction across all 6 arm pairs x 2 endpoints. Positive C/D favors full duplex.",
             fontsize=9, linespacing=1.5)
    fig.subplots_adjust(left=.15, right=.98, top=.85, bottom=.18, hspace=.61, wspace=.46)
    for suffix in ("pdf", "png"):
        fig.savefig(output / f"LOGIC5_COMMUNICATION_STATISTICS.{suffix}", dpi=220)
    plt.close(fig)


def analyze(root, output):
    rows = extract(root)
    tests = statistics(rows)
    output.mkdir(parents=True, exist_ok=True)
    summary = {arm: {"n": 20, "median_loss": float(np.median(values(rows, arm, "loss"))),
                       "median_qualified": float(np.median(values(rows, arm, "qualified"))),
                       "mean_qualified": float(np.mean(values(rows, arm, "qualified")))} for arm in ARMS}
    result = {"root": str(root), "plan_sha256": sha(root / "plan.json"), "analysis_sha256": sha(Path(__file__)),
        "analysis_type": "post_hoc", "summary": summary, "endpoints": rows, "tests": tests,
        "methods": "Two-sided exact sign-permutation Wilcoxon signed-rank, Pratt zeros, loss ties rtol=1e-9/atol=1e-12; exact finite rank-sum distribution via dynamic programming (validated against exhaustive SciPy permutations). Holm over 12 tests. Paired percentile bootstrap 20000 seed resamples, RNG 20260924. Tests use raw differences; loss effect is median paired ratio. Individual bootstrap CIs are not simultaneous intervals."}
    (output / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    write_csv(output / "endpoints.csv", rows)
    write_csv(output / "paired_tests.csv", tests)
    plot(output, rows, tests)
    lines = ["# Logic-5 Four-Arm Paired Endpoint Analysis", "", "## Scope and Methods", "",
        "All 80 endpoints (20 matched seeds per arm), completed at 100 epochs. Saved terminal_union.json records were checked against completed.json; manifests, budgets, drained streams and provider-balance histories were audited. No simulations or model calls. Structural hashes, not rate vectors, define topology identity in the existing union builder. This analysis does not independently re-enumerate the databases.", "",
        result["methods"], "", "The signed-rank test assumes symmetric paired differences under its null. Exact enumeration handles ties but does not remove that assumption. Bootstrap intervals describe effect uncertainty, not standard deviation or simultaneous significance. These are post-hoc analyses, not preregistered confirmatory tests.", "",
        "## Endpoint Summaries", "", "| Arm | Median loss | Median qualifying count | Mean qualifying count |",
        "|---|---:|---:|---:|"]
    for arm, v in summary.items():
        lines.append(f"| {LABELS[arm]} | {v['median_loss']:.7g} | {v['median_qualified']:.1f} | {v['mean_qualified']:.2f} |")
    lines += ["", "## All Paired Tests", "", "W/T/L favors the first arm (lower loss, higher count). All comparisons are included, regardless of direction or significance.", "",
        "| First arm vs second | Endpoint | W/T/L | Effect | Bootstrap 95% CI | Raw p | Holm p (12) |",
        "|---|---|---:|---:|---|---:|---:|"]
    for t in tests:
        if t["metric"] == "loss":
            effect = 100*(1-t["median_paired_ratio"])
            ci = [100*(1-t["median_ratio_ci95"][1]), 100*(1-t["median_ratio_ci95"][0])]
            label = f"{effect:.1f}% median paired reduction"
        else:
            effect, ci = t["mean_difference"], t["mean_difference_ci95"]
            label = f"{effect:+.2f} mean additional topologies"
        lines.append(f"| {LABELS[t['a']]} vs {LABELS[t['b']]} | {t['metric']} | {t['wins']}/{t['ties']}/{t['losses']} | {label} | [{ci[0]:.2f}, {ci[1]:.2f}] | {t['p']:.6g} | {t['holm_p_12']:.6g} |")
    lines += ["", "## Main Findings", ""]
    for t in tests:
        if t["a"] == "full" and t["metric"] == "loss":
            ci = [100*(1-t["median_ratio_ci95"][1]), 100*(1-t["median_ratio_ci95"][0])]
            lines.append(f"- Full duplex vs {LABELS[t['b']]}: {100*(1-t['median_paired_ratio']):.1f}% median paired loss reduction, bootstrap 95% interval [{ci[0]:.1f}%, {ci[1]:.1f}%], Holm p={t['holm_p_12']:.6g}.")
    for t in tests:
        if t["a"] == "full" and t["metric"] == "qualified":
            lines.append(f"- Full duplex vs {LABELS[t['b']]}: {t['mean_difference']:+.2f} mean qualifying-topology difference, Holm p={t['holm_p_12']:.6g}.")
    lines += ["", "These results support better rank-one loss for the full communication system at this horizon, accompanied by a qualification-count trade-off versus insertion-only. They do not support a general increase in topology yield, nor by themselves isolate a causal mechanism or establish a formal factorial interaction.", "", "## Interpretation and Limits", "",
        "Read corrected p-values alongside effect sizes. A difference between group medians is not the median within-seed effect. Failure to reject a difference is not equivalence. Counts use the frozen loss threshold (0.04271267898491677), not full truth-table correctness or a formal structural-distance diversity measure. Threshold-dependent quality/yield curves and classification diagnostics remain separate analyses.", "",
        "This compares complete search systems: RL, mutation, random exploration, AdaGrad and LLM proposals. No-communication also includes both search and evaluator-guided LLM self-feedback, pooled only at the end. Effects cannot be attributed solely to elite imitation or isolated RL learning. Seeds are matched, but proposal banks, completed request counts, actual simulation expenditures and asynchronous insertion schedules are not identical.", "",
        "Seeds 1-9 retain original completed runs; seeds 10-12 replace all outage-interrupted arms from epoch zero; seeds 13-20 are fresh first attempts. Failed original attempts were not selected by performance and remain excluded from endpoints, with costs retained separately. Storage and wall-clock scheduling differ across these blocks. "
        f"{sum(r['failed_requests'] for r in rows)} ordinary failed requests across {sum(r['requests'] for r in rows)} recorded request histories remain outcomes.", "",
        "Artifacts: LOGIC5_COMMUNICATION_STATISTICS.pdf / .png; endpoints.csv; paired_tests.csv; summary.json (input paths and hashes).", "",
        f"Reproduce: `.venv/bin/python -m comparisons.search_exploration.analyze_logic5_four_arm --root {root} --output {output}`", ""]
    (output / "REPORT.md").write_text("\n".join(lines))
    print(json.dumps({"summary": summary, "tests": tests}, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    analyze(args.root, args.output)
