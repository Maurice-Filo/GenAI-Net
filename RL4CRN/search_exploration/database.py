"""Existing SQLite schema with explicit new-source provenance, without legacy edits."""

import json
from RL4CRN.utils.results_database import ResultsDatabase, _task_info_dumps


class ExplorationResultsDatabase(ResultsDatabase):
    def _enqueue(self, kind, payload):
        if kind == "hof_snapshot":
            for entry in payload["entries"]:
                info = json.loads(entry["task_info_json"])
                source = info.get("exploration_source")
                if source:
                    entry["emitter"] = source.upper()
                    entry["provenance_class"] = "direct_llm" if source == "llm" else source + "_seed_refined"
                    info.update(emitter=entry["emitter"], provenance_class=entry["provenance_class"])
                    entry["task_info_json"] = _task_info_dumps(info)
        return super()._enqueue(kind, payload)
