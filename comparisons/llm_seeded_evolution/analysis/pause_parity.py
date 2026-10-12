"""Explicit, identity-checked in-memory suspension of the parity process trees."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from comparisons.llm_seeded_evolution.replay import GENERATED


def info(pid):
    path = Path(f"/proc/{pid}")
    try:
        text = (path / "stat").read_text()
        rest = text.rsplit(")", 1)[1].split()
        return {"pid": pid, "ppid": int(rest[1]), "starttime": rest[19], "state": rest[0],
                "uid": path.stat().st_uid, "name": text.split("(", 1)[1].rsplit(")", 1)[0]}
    except FileNotFoundError:
        return None


def identical(row):
    current = info(row["pid"])
    return current and current["starttime"] == row["starttime"] and current["uid"] == os.getuid()


def snapshot():
    return {p: r for directory in Path('/proc').iterdir() if directory.name.isdigit()
            if (p := int(directory.name)) and (r := info(p)) and r["uid"] == os.getuid()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.resume:
        record = json.loads(args.record.read_text())
        for row in record["processes"]:
            if not identical(row):
                raise RuntimeError("Saved process identity no longer matches; inspect before resuming")
        for row in reversed(record["processes"]):
            os.kill(row["pid"], signal.SIGCONT)
        record["status"] = "resumed_requires_timeout_review"
        args.record.write_text(json.dumps(record, indent=2) + "\n")
        return
    if args.record.exists():
        raise FileExistsError("Pause record exists; no overwrite")
    roots = []
    for session, marker in (("parity12-workspace-loss", "run_parity12_communication_queue"),
                            ("ga-parity12-followup", "parity12.queue")):
        pid = int(subprocess.check_output(["tmux", "display-message", "-p", "-t", session, "#{pane_pid}"], text=True))
        row = info(pid)
        command = Path(f"/proc/{pid}/cmdline").read_bytes().decode(errors="replace")
        if not row or row["uid"] != os.getuid() or marker not in command:
            raise ValueError(f"Unexpected process in {session}")
        roots.append(row)
    record = {"status": "suspending", "created_at": datetime.now(timezone.utc).isoformat(),
        "reason": "User paused long parity runs for hyperparameter review; separate short GA trial requested",
        "warning": "SIGSTOP preserves RAM, not a durable checkpoint; GPU allocations remain; wall-clock/provider timeouts need review before resume",
        "roots": roots, "processes": [], "campaign_status_snapshots": {}}
    schedule = json.loads((GENERATED / "parity12_communication_300epoch_20260909_schedule.json").read_text())
    for arm, entry in schedule["campaigns"].items():
        path = Path(entry["audit_command"][2]) / "status.json"
        record["campaign_status_snapshots"][arm] = json.loads(path.read_text())
    args.record.parent.mkdir(parents=True, exist_ok=True)
    def save():
        args.record.write_text(json.dumps(record, indent=2) + "\n")
    save()
    seen = set()
    frontier = roots
    while frontier:
        for row in frontier:
            if row["pid"] in seen:
                continue
            if identical(row):
                os.kill(row["pid"], signal.SIGSTOP)
                record["processes"].append(row)
                seen.add(row["pid"])
        save()
        table = snapshot()
        frontier = [row for pid, row in table.items() if pid not in seen and row["ppid"] in seen]
    time.sleep(.5)
    # tmux resumes its pane leaders; the child campaign launchers and workers stay stopped.
    root_ids = {r["pid"] for r in roots}
    remaining = [row["pid"] for row in record["processes"] if row["pid"] not in root_ids
                 and identical(row) and info(row["pid"])["state"] not in ("T", "t", "Z")]
    record.update(status="suspended" if not remaining else "suspension_incomplete", not_stopped=remaining,
                  pane_leaders_may_be_resumed_by_tmux=True,
                  verified_at=datetime.now(timezone.utc).isoformat())
    save()
    if remaining:
        raise RuntimeError(f"Some processes are not stopped: {remaining}")
    print(json.dumps({"status": record["status"], "processes": len(record["processes"]), "record": str(args.record)}))


if __name__ == "__main__":
    main()
