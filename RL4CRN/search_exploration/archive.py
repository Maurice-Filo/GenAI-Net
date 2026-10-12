"""Bounded in-memory discovery elites; all fitted candidates remain on disk."""

import heapq
import math

import numpy as np


def descriptor(ids, library, output_labels, kind, topology_hash):
    if kind == "topology":
        return topology_hash
    # Cheap pilot descriptor, not a mechanistic-equivalence claim.
    output = set(output_labels)
    reactions = [library[i] for i in ids]
    producing = sum(any(r.get_stoichiometry_dict().get(s, 0) > 0 for s in output) for r in reactions)
    consuming = sum(any(r.get_stoichiometry_dict().get(s, 0) < 0 for s in output) for r in reactions)
    catalytic = sum(bool(set(r.reactant_labels) & set(r.product_labels)) for r in reactions)
    source = sum(not r.reactant_labels for r in reactions)
    sink = sum(not r.product_labels for r in reactions)
    species = len(set().union(*(set(r.reactant_labels + r.product_labels) for r in reactions)))
    return ":".join(map(str, (producing, consuming, catalytic, source, sink, species)))


class DiscoveryArchive:
    def __init__(self, config):
        self.cfg = config
        self.entries = {}
        self.heap = []
        self.revision = 0
        self._ranked = None
        self.region_best = {}

    def utility(self, record):
        if not record.get("qualified", False):
            return 0.
        quality = max(0., min(1., 1 - record["loss"] / self.cfg.quality_threshold))
        return 1 + self.cfg.quality_improvement_weight * quality

    def add(self, record, store):
        store.save_design(record)
        region = record["region"]
        previous_visits = store.count("region:" + region)
        store.count("region:" + region, 1)
        store.count("topology:" + record["topology_hash"], 1)
        old = self.region_best.get(region)
        gain = max(0., self.utility(record) - (self.utility(old) if old else 0.))
        if record["simulation_complete"] and (old is None or self.utility(record) > self.utility(old)
                or (self.utility(record) == self.utility(old) and record["loss"] < old["loss"])):
            # Region summaries are compact and persist even if a topology leaves the live archive.
            self.region_best[region] = {k: record[k] for k in ("loss", "qualified", "topology_hash")}
            if old and old["topology_hash"] in self.entries and old["topology_hash"] != record["topology_hash"]:
                former = self.entries[old["topology_hash"]]
                heapq.heappush(self.heap, (self._eviction_priority(former), former["revision"], former["topology_hash"]))
        gate = math.exp(-min(700., max(0., record["loss"]) / self.cfg.quality_threshold)) if record["simulation_complete"] else 0.
        novelty = gate / math.sqrt(1 + previous_visits)
        key = record["topology_hash"]
        incumbent = self.entries.get(key)
        if record["simulation_complete"] and (incumbent is None or record["loss"] < incumbent["loss"]):
            self.revision += 1
            # Cache payloads contain trajectories; live elite records intentionally do not.
            compact = {k: v for k, v in record.items() if k != "task_info"}
            compact["revision"] = self.revision
            self.entries[key] = compact
            heapq.heappush(self.heap, (self._eviction_priority(compact), self.revision, key))
            while len(self.entries) > self.cfg.archive_capacity:
                priority, revision, remove = heapq.heappop(self.heap)
                if remove in self.entries and self.entries[remove]["revision"] == revision:
                    current = self._eviction_priority(self.entries[remove])
                    if priority != current:
                        heapq.heappush(self.heap, (current, revision, remove))
                        continue
                    del self.entries[remove]
            if len(self.heap) > 3 * self.cfg.archive_capacity:
                self.heap = [(self._eviction_priority(r), r["revision"], k) for k, r in self.entries.items()]
                heapq.heapify(self.heap)
            self._ranked = None
        return gain, novelty

    def _eviction_priority(self, record):
        representative = self.region_best.get(record["region"], {}).get("topology_hash") == record["topology_hash"]
        return (int(representative), -self.utility(record), -record["loss"])

    def ranked(self, limit=None):
        if self._ranked is None:
            self._ranked = sorted(self.entries.values(), key=lambda r: (r["loss"], r["topology_hash"]))
        return self._ranked if limit is None else self._ranked[:limit]

    def imitation(self, rng):
        top = self.ranked(self.cfg.sil_top_k)
        if not top:
            return []
        if not self.cfg.sil_region_balanced:
            indices = rng.choice(len(top), min(len(top), self.cfg.sil_batch_size), replace=False)
            return [top[i] for i in indices]
        groups = {}
        for entry in top:
            groups.setdefault(entry["region"], []).append(entry)
        regions = sorted(groups)
        return [groups[region][int(rng.integers(len(groups[region])))]
                for region in rng.choice(regions, size=min(len(top), self.cfg.sil_batch_size), replace=True)]

    def parent(self, rng, store):
        return self.parent_sampler(store)(rng)

    def parent_sampler(self, store):
        """Freeze grouping and visits for a cohort; preserve draw ordering."""
        grouped = {}
        for entry in self.entries.values():
            grouped.setdefault(entry["region"], []).append(entry)
        if not grouped:
            return lambda rng: None
        keys = sorted(grouped)
        counts = store.counts(["region:" + r for r in keys])
        weights = np.array([1 / math.sqrt(1 + counts["region:" + r]) for r in keys])
        probabilities = weights / weights.sum()
        def draw(rng):
            region = keys[int(rng.choice(len(keys), p=probabilities))]
            choices = grouped[region]
            return choices[int(rng.integers(len(choices)))]
        return draw

    def state_dict(self):
        return {"entries": self.entries, "region_best": self.region_best, "revision": self.revision}

    def load_state_dict(self, state):
        self.entries, self.region_best, self.revision = state["entries"], state["region_best"], state["revision"]
        self.heap = [(self._eviction_priority(r), r["revision"], k) for k, r in self.entries.items()]
        heapq.heapify(self.heap)
        self._ranked = None
