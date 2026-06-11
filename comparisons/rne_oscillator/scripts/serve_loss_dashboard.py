#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import socket
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse


ROOT = Path(__file__).resolve().parents[3]
RAW_ROOT = ROOT / "comparisons/rne_oscillator/data/raw"
DEFAULT_METHOD = "rl4crn_rne_oscillator_trace_8rxn_stop_risksched085to09_lr1e4_entropy005_s4_c1_rnetol_d256_depth3_p20"


def _methods() -> list[str]:
    if not RAW_ROOT.exists():
        return []
    return sorted(p.name for p in RAW_ROOT.iterdir() if p.is_dir() and p.name != "raw")


def _read_method(method: str) -> dict:
    method_dir = RAW_ROOT / method
    traces = []
    latest = []
    success_count = 0
    finished_count = 0

    for progress_path in sorted(method_dir.glob("seed*/progress.csv")):
        seed = progress_path.parent.name
        rows = list(csv.DictReader(progress_path.open(newline="", encoding="utf-8")))
        if not rows:
            continue

        epochs = [int(float(row["epoch"])) for row in rows]
        losses = [float(row["saved_best_loss"]) for row in rows]
        epoch_losses = [float(row["epoch_best_loss"]) for row in rows]
        success = any(str(row.get("success", "")).lower() == "true" for row in rows)
        finished = (progress_path.parent / "result.json").exists()
        if finished:
            finished_count += 1
        if success:
            success_count += 1

        traces.append(
            {
                "seed": seed,
                "epochs": epochs,
                "losses": losses,
                "epoch_losses": epoch_losses,
                "success": success,
                "finished": finished,
            }
        )
        latest.append(
            {
                "seed": seed,
                "epoch": epochs[-1],
                "loss": losses[-1],
                "epoch_loss": epoch_losses[-1],
                "success": success,
                "finished": finished,
            }
        )

    latest.sort(key=lambda row: row["seed"])
    best = min((row["loss"] for row in latest), default=None)
    seeds = [row["seed"] for row in latest]
    return {
        "method": method,
        "methods": _methods(),
        "success_threshold": 20.0,
        "finished_count": finished_count,
        "success_count": success_count,
        "active_count": len(latest) - finished_count,
        "trace_count": len(latest),
        "first_seed": seeds[0] if seeds else None,
        "last_seed": seeds[-1] if seeds else None,
        "best_loss": best,
        "traces": traces,
        "latest": latest,
    }


HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>RNE Oscillator Loss Monitor</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    body { margin: 0; font-family: system-ui, -apple-system, Segoe UI, sans-serif; background: #fafafa; color: #1f2933; }
    header { display: flex; gap: 16px; align-items: center; padding: 14px 18px; border-bottom: 1px solid #d8dee4; background: #fff; }
    h1 { font-size: 18px; margin: 0; font-weight: 650; }
    select { font: inherit; padding: 4px 8px; }
    #stats { margin-left: auto; font-size: 13px; color: #4b5563; white-space: nowrap; }
    #plot { width: 100vw; height: calc(100vh - 60px); }
  </style>
</head>
<body>
  <header>
    <h1>RNE oscillator losses</h1>
    <select id="method"></select>
    <div id="stats">Loading...</div>
  </header>
  <div id="plot"></div>
<script>
const defaultMethod = "__METHOD__";
let currentMethod = new URLSearchParams(location.search).get("method") || defaultMethod;

function label(row) {
  let s = row.seed;
  if (row.success) s += " success";
  else if (!row.finished) s += " active";
  return s;
}

async function refresh() {
  const res = await fetch(`/data?method=${encodeURIComponent(currentMethod)}&t=${Date.now()}`);
  const data = await res.json();
  const select = document.getElementById("method");
  if (!select.options.length) {
    for (const method of data.methods) {
      const opt = document.createElement("option");
      opt.value = method;
      opt.textContent = method;
      if (method === currentMethod) opt.selected = true;
      select.appendChild(opt);
    }
  }

  const traces = data.traces.map((row) => ({
    x: row.epochs,
    y: row.losses,
    type: "scatter",
    mode: "lines",
    name: label(row),
    line: { width: row.success ? 4 : 2 },
    opacity: row.finished ? 0.78 : 0.95
  }));
  traces.push({
    x: [0, 801],
    y: [data.success_threshold, data.success_threshold],
    type: "scatter",
    mode: "lines",
    name: "success threshold",
    line: { color: "black", dash: "dash", width: 2 }
  });

  const bestText = data.best_loss === null ? "n/a" : data.best_loss.toFixed(4);
  const seedSpan = data.trace_count ? `${data.first_seed}-${data.last_seed}` : "n/a";
  document.getElementById("stats").textContent =
    `traces ${data.trace_count} (${seedSpan}) | finished ${data.finished_count} | successes ${data.success_count} | active ${data.active_count} | best ${bestText}`;

  Plotly.react("plot", traces, {
    title: data.method,
    xaxis: { title: "Epoch", range: [0, 801], gridcolor: "#e2e8f0" },
    yaxis: { title: "Best saved loss so far", rangemode: "tozero", gridcolor: "#e2e8f0" },
    margin: { l: 70, r: 25, t: 50, b: 55 },
    paper_bgcolor: "#fafafa",
    plot_bgcolor: "#ffffff",
    legend: { orientation: "h", y: 1.08, x: 0 }
  }, { responsive: true });
}

document.getElementById("method").addEventListener("change", (event) => {
  currentMethod = event.target.value;
  history.replaceState(null, "", `/?method=${encodeURIComponent(currentMethod)}`);
  refresh();
});

refresh();
setInterval(refresh, 10000);
</script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--method", default=DEFAULT_METHOD)
    args = parser.parse_args()

    default_method = args.method

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            parsed = urlparse(self.path)
            if parsed.path == "/data":
                qs = parse_qs(parsed.query)
                method = qs.get("method", [default_method])[0]
                payload = json.dumps(_read_method(method)).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return

            payload = HTML.replace("__METHOD__", default_method).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, fmt, *args):
            print(f"{self.client_address[0]} - {fmt % args}", flush=True)

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    host, port = server.server_address
    print(f"Serving RNE oscillator dashboard on http://{host}:{port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
