"""One persistent Julia process; whole optimization jobs cross the IPC boundary."""

from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import queue
import subprocess
import tempfile
import threading

import numpy as np


class JuliaOptimizer:
    def __init__(self, *, project=None, julia="julia", startup_timeout=180., cache_size=32):
        project = project or os.environ.get("SYNTH_EVO_PROJECT")
        if not project or not (Path(project) / "src/CRNOptim.jl").is_file():
            raise ValueError("Set project/SYNTH_EVO_PROJECT to the SynthEvo Julia package directory")
        if not isinstance(cache_size, int) or cache_size < 1:
            raise ValueError("cache_size must be a positive integer")
        self._lock = threading.RLock()
        self._models = OrderedDict()
        self._cache_size = cache_size
        self._closed = False
        self._responses = queue.Queue()
        self._stderr = tempfile.TemporaryFile(mode="w+t")
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="julia-rates")
        env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", JULIA_NUM_THREADS="1")
        script = Path(__file__).with_name("worker.jl")
        self.command = [julia, f"--project={Path(project).resolve()}", "--startup-file=no",
                        "--threads=1", str(script)]
        try:
            self.process = subprocess.Popen(self.command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                            stderr=self._stderr, text=True, bufsize=1, env=env)
            self._reader = threading.Thread(target=self._read, daemon=True, name="julia-rates-reader")
            self._reader.start()
            self.info = self._receive(startup_timeout)
            if self.info.get("status") != "ready":
                raise RuntimeError(f"Julia startup failed: {self.info}")
            self._backend = {k: self.info[k] for k in ("julia", "sundials", "scimlbase",
                             "optimizer_source_sha256", "worker_sha256", "manifest_sha256",
                             "feasible_source_sha256")}
        except BaseException:
            self.close()
            raise

    def _read(self):
        try:
            for line in self.process.stdout:
                try:
                    self._responses.put(json.loads(line))
                except ValueError:
                    self._responses.put(RuntimeError(f"Unexpected Julia protocol output: {line[:500]}"))
        finally:
            self._responses.put(EOFError("Julia worker exited"))

    def _receive(self, timeout):
        try:
            result = self._responses.get(timeout=timeout)
        except queue.Empty:
            self._stop_process()
            raise TimeoutError("Julia request deadline exceeded; worker stopped, no result substituted") from None
        if isinstance(result, BaseException):
            self._stderr.seek(0)
            detail = self._stderr.read()[-8000:]
            self._stop_process()
            raise RuntimeError(f"{result}\n{detail}")
        if result.get("status") == "error":
            raise RuntimeError(result["error"])
        return result

    def _request(self, command, *, timeout=120., **payload):
        if timeout is not None and (not np.isfinite(timeout) or timeout <= 0):
            raise ValueError("timeout must be positive and finite, or None")
        with self._lock:
            if self._closed or self.process.poll() is not None:
                raise RuntimeError("Julia worker is closed; explicitly create a new worker")
            message = json.dumps({"command": command, **payload}, allow_nan=False, separators=(",", ":"))
            self.process.stdin.write(message + "\n")
            self.process.stdin.flush()
            return self._receive(timeout)

    def prepare(self, crn, experiment):
        payload = {"crn": crn.payload(), "experiment": experiment.payload(crn)}
        key = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        with self._lock:
            if self._closed or self.process.poll() is not None:
                raise RuntimeError("Julia worker is closed")
            if key not in self._models:
                self._request("prepare", key=key, **payload)
                self._models[key] = crn.parameter_count
                if len(self._models) > self._cache_size:
                    old, _ = self._models.popitem(last=False)
                    self._request("release", key=old)
            self._models.move_to_end(key)
        return key

    def _parameter_count(self, model):
        if model not in self._models:
            raise ValueError("Unknown/evicted model handle; prepare it again")
        self._models.move_to_end(model)
        return self._models[model]

    def evaluate(self, model, rates, *, trajectories=False, timeout=120.):
        with self._lock:
            p = np.asarray(rates, dtype=float)
            if p.shape != (self._parameter_count(model),) or not np.isfinite(p).all() or np.any(p <= 0):
                raise ValueError("Expected one positive finite rate per free reaction")
            return self._request("evaluate", key=model, rates=p.tolist(),
                                 trajectories=bool(trajectories), timeout=timeout)

    def optimize(self, model, starts=None, *, states=None, steps=32, method="adagrad",
                 learning_rate=.1, bounds=(.001, 100.), timeout=120.,
                 step_policy="stop", max_backtracks=8, max_evaluations=None,
                 evaluation_limit=None):
        if method not in ("adagrad", "adam"):
            raise ValueError("method must be adagrad or adam")
        if not isinstance(steps, int) or steps < 0:
            raise ValueError("steps must be a nonnegative integer")
        lo, hi = map(float, bounds)
        if not (np.isfinite([lo, hi, learning_rate]).all() and 0 < lo < hi and learning_rate > 0):
            raise ValueError("Invalid bounds/learning_rate")
        config = dict(method=method, learning_rate=float(learning_rate), bounds=[lo, hi])
        if step_policy not in ("stop", "feasible"):
            raise ValueError("step_policy must be stop or feasible")
        if not isinstance(max_backtracks, int) or not 0 <= max_backtracks <= 30:
            raise ValueError("max_backtracks must be an integer from 0 to 30")
        if max_evaluations is not None and (not isinstance(max_evaluations, int) or max_evaluations < 1):
            raise ValueError("max_evaluations must be a positive integer or None")
        if step_policy == "stop" and max_evaluations is not None:
            raise ValueError("max_evaluations currently requires step_policy=feasible")
        if evaluation_limit is not None and (step_policy != "feasible" or
                type(evaluation_limit) is not int or evaluation_limit < 1):
            raise ValueError("evaluation_limit requires feasible policy and a positive integer")
        if step_policy == "feasible":
            config.update(step_policy=step_policy, max_backtracks=max_backtracks,
                          max_evaluations=max_evaluations)
        if (starts is None) == (states is None):
            raise ValueError("Supply exactly one of starts or states")
        with self._lock:
            n = self._parameter_count(model)
            if n == 0:
                raise ValueError("No free parameters to optimize")
            if starts is not None:
                starts = np.asarray(starts, dtype=float)
                if starts.ndim != 2 or starts.shape[1] != n or not len(starts):
                    raise ValueError("starts must be [restart,free_parameter]")
                # Invalid starts are represented explicitly, without discarding valid siblings.
                starts = [[float(v) if np.isfinite(v) else None for v in p] for p in starts]
            else:
                if not states:
                    raise ValueError("states must be nonempty")
                for state in states:
                    if state.get("key") != model or state.get("config") != config:
                        raise ValueError("Optimizer state/model/configuration mismatch")
                    if state.get("backend") != self._backend or state.get("checksum") != self._state_checksum(state):
                        raise ValueError("Optimizer checkpoint was modified or backend has changed")
            result = self._request("optimize", key=model, starts=starts, states=states,
                                   steps=steps, config=config, evaluation_limit=evaluation_limit,
                                   timeout=timeout)
            for item in result["results"]:
                if "state" in item:
                    item["state"]["backend"] = dict(self._backend)
                    item["state"]["checksum"] = self._state_checksum(item["state"])
            return result

    @staticmethod
    def _state_checksum(state):
        payload = {k: v for k, v in state.items() if k != "checksum"}
        return hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode()).hexdigest()

    def submit(self, model, starts=None, **kwargs):
        """Queue an entire optimization job without blocking the caller/search."""
        return self._executor.submit(self.optimize, model, deepcopy(starts), **deepcopy(kwargs))

    def stats(self):
        return self._request("stats")

    def _stop_process(self):
        process = getattr(self, "process", None)
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()

    def close(self):
        if self._closed:
            return
        self._closed = True
        self._stop_process()
        self._executor.shutdown(wait=True, cancel_futures=True)
        if hasattr(self, "_reader"):
            self._reader.join(timeout=3)
        if hasattr(self, "process"):
            self.process.stdin.close()
            self.process.stdout.close()
        self._stderr.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
