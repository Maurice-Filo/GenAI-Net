# %% [markdown]
# # GenAI-Net (RL4CRN) — Unified Tasks + Training Tutorial
#
# This notebook **merges**:
# 1) the **Task Presets / TaskSpec factory** tutorial (logic, tracking, oscillator, dose-response, SSA)
# 2) the **minimal training** tutorial (session + trainer, stop/resume, checkpoints)
#
# The goal is to **exercise the full stack**:
# - task construction (`make_task`)
# - session wiring (`make_session_and_trainer`)
# - training loop (`trainer.run`, `trainer.step_epoch`)
# - inspection (`trainer.inspect_best`)
# - checkpoint save/load
#
# ---
# ## 0) Environment setup
#
# From repo root:
# ```bash
# pip install -e .
# ```

# %%
# %% [setup]
import os
import sys
import numpy as np

print("Python:", sys.version.split()[0])
print("CWD:", os.getcwd())

# %% [markdown]
# ---
# ## 1) Imports
#
# We use the **standard interface layer** in `RL4CRN/utils/input_interface.py`:
# - `Configurator`: presets + overrides
# - `make_task`: standardized task builder
# - `make_session_and_trainer`: full wiring (template/library/env/interfaces/policy/agent/trainer)

# %%
# %% [imports]
from RL4CRN.utils.input_interface import (
    Configurator,
    make_task,
    make_session_and_trainer,
)

from RL4CRN.utils.visualizations import plot_truth_table

# %% [markdown]
# ---
# ## 2) Helper utilities (notebook-local)
#
# A few tiny helpers for clean display and consistent smoke tests.

# %%
# %% [helpers]
def print_task_summary(task, max_preview=3):
    """Print a compact TaskSpec summary."""
    print("Task:", task.name)
    print("time_horizon:", task.time_horizon.shape, f"[0..{task.time_horizon[-1]}]")
    print("num scenarios:", len(task.u_list))
    if len(task.u_list) > 0:
        print(f"first {min(max_preview, len(task.u_list))} u:", task.u_list[:max_preview])
    print()


def run_smoke_reward(task, state, label=""):
    """Call task.compute_reward on a given state and print normalized output."""
    out = task.compute_reward(state)
    if isinstance(out, tuple):
        loss, info = out
    else:
        loss, info = out, {}
    print(f"[reward smoke{(' - ' + label) if label else ''}] loss={float(loss):.6g} | info_keys={list(info.keys())[:8]}")
    return out

# %% [markdown]
# ---
# ## 3) Task factory demo (no training yet)
#
# This section mirrors the "Task Presets Tutorial" idea: construct tasks with a few knobs.
# Here we build tasks **standalone** (no environments), just to show the API and validate object shapes.

# %%
# %% [task-scaffold]
species_labels_logic = [f"X_{i+1}" for i in range(3)] + ["OUT"]
species_labels_rpa   = ["X_1", "X_2", "X_3", "X_4", "X_5", "X_6", "OUT"]
species_labels_osc   = ["X_1", "X_2", "X_3"]
species_labels_dose  = ["X_1", "X_2", "OUT"]

print("Logic labels:", species_labels_logic)
print("RPA labels:", species_labels_rpa)
print("Osc labels:", species_labels_osc)
print("Dose labels:", species_labels_dose)

# %% [markdown]
# ### 3.1) Logic task

# %%
# %% [logic-task]
logic_fn = lambda u: (bool(u[0]) and (not bool(u[1]))) or bool(u[2])

task_logic = make_task(
    kind="logic",
    species_labels=species_labels_logic,
    n_inputs=3,
    logic_fn=logic_fn,
    ic="zero",
    weights="steady_state",
    t_f=100, n_t=200,
)
print_task_summary(task_logic)

# Plot truth table target (we reconstruct r_list for visualization)
r_list = [np.array([float(bool(logic_fn(u)))], dtype=np.float32) for u in task_logic.u_list]
plot_truth_table(task_logic.u_list, r_list, title="Target truth table (logic task)")

# %% [markdown]
# ### 3.2) Deterministic tracking (RPA-style)

# %%
# %% [tracking-task]
task_tracking = make_task(
    kind="tracking",
    species_labels=species_labels_rpa,
    p=3,
    u_values=[0.5, 1.0, 1.5],
    target="copy_input0",
    ic=("constant", 0.01),
    weights="transient",
    t_f=100, n_t=200,
)
print_task_summary(task_tracking)

# %% [markdown]
# ### 3.3) Oscillator discovery (mean-level targeting)

# %%
# %% [oscillator-task]
task_osc = make_task(
    kind="oscillator",
    species_labels=species_labels_osc,
    p=1,
    u_values=[1.0],
    ic=("constant", 0.01),
    t_f=100, n_t=200,
    osc_w=[0.4, 0.0, 0.2, 0.4],
    t0=20.0,
)
print_task_summary(task_osc)

# %% [markdown]
# ### 3.4) Dose–response matching (target is a function)

# %%
# %% [dose-response-task]
def hill_function(u: float, kd=5.0, max_production=50.0, n=6.0) -> float:
    return max_production * (u**n) / (kd**n + u**n)

task_dose = make_task(
    kind="dose_response",
    species_labels=species_labels_dose,
    dose_range=(0.0, 10.0, 10),
    target_fn=hill_function,
    ic=("constant", 0.1),
    weights="transient",
    t_f=100, n_t=200,
)
print_task_summary(task_dose)

# %% [markdown]
# ### 3.5) Stochastic SSA tracking (smoke only)
#
# These can be heavier; here we only build the task object and keep `n_t` small.

# %%
# %% [ssa-tracking-task]
task_ssa = make_task(
    kind="ssa_tracking",
    species_labels=species_labels_rpa,
    p=2,
    u_values=[1.0, 2.0, 3.0],
    target="copy_input0",
    ic="zero",
    weights="steady_state",
    t_f=50, n_t=100,
    n_trajectories=64,
    max_threads=1024,
)
print_task_summary(task_ssa)

# %% [markdown]
# ### 3.6) Robust SSA (smoke only)

# %%
# %% [ssa-robust-task]
task_ssa_robust = make_task(
    kind="ssa_robust",
    species_labels=species_labels_rpa,
    p=2,
    u_values=[1.0, 2.0, 3.0],
    target="copy_input0",
    ic="zero",
    weights="steady_state",
    t_f=50, n_t=100,
    n_trajectories=64,
    max_threads=1024,
    rpa_weight=3.0,
    cv_weight=1.0,
)
print_task_summary(task_ssa_robust)

# %% [markdown]
# ---
# ## 4) Full wiring + training (logic task)
#
# This section mirrors the **older training notebook**, but uses the updated session object:
# - `session.task.u_list` instead of `session.u_list`
# - `session.task.compute_reward` is used by the environment via `Trainer`

# %%
# %% [user-config-logic-training]
cfg = Configurator.preset("fast")

# Task config
cfg.task.kind = "logic"
cfg.task.n_inputs = 3
cfg.task.logic_fn = lambda u: (bool(u[0]) and (not bool(u[1]))) or bool(u[2])
cfg.task.t_f = 100.0
cfg.task.N_t = 200
cfg.task.weights = "steady_state"
cfg.task.ic_value = 0.01

# Train config
cfg.train.max_added_reactions = 5
cfg.train.epochs = 30
cfg.train.render_every = 5
cfg.train.seed = 0

# Library / solver
cfg.library.order = 2
cfg.library.include_dilution = False
cfg.solver.algorithm = "CVODE"
cfg.solver.rtol = 1e-10
cfg.solver.atol = 1e-10

# %%
# %% [build-session]
session, trainer = make_session_and_trainer(cfg, device="auto")

print(f"Device: {session.device}")
print(f"CPUs: {session.n_cpus} | Batch size: {session.batch_size}")
print(f"Task: {session.task.name} | Inputs: {cfg.task.n_inputs} | Max steps: {cfg.train.max_added_reactions}")
print()

# Plot target truth table
u_list = session.task.u_list
r_list = [np.array([float(bool(cfg.task.logic_fn(u)))], dtype=np.float32) for u in u_list]
plot_truth_table(u_list, r_list, title="Target truth table (session task)")

print("Template IO-CRN:")
print(session.crn_template)

# %% [markdown]
# ### 4.1) Reward smoke test on template state
#
# This validates the end-to-end reward signature on an actual CRN object.

# %%
# %% [reward-smoke-template]
run_smoke_reward(session.task, session.crn_template, label="template")

# %% [markdown]
# ### 4.2) Train (stop/resume friendly)
#
# - `trainer.run(...)` supports Ctrl+C
# - pass `checkpoint_path` to save periodically and on interrupt

# %%
# %% [train-logic]
checkpoint_path = "logic_chkpt.pkl"
trainer.run(epochs=cfg.train.epochs, checkpoint_path=checkpoint_path)

# %%
# %% [inspect-best-logic]
trainer.inspect_best(plot=True)

# %%
# %% [final-inspection-logic]
best = trainer.best_crn()
print("Hall of Fame size:", len(session.mult_env.hall_of_fame))
if best is not None:
    print("Best loss:", best.last_task_info.get("reward", None))

# %% [markdown]
# ### 4.3) Checkpoint load (resume workflow)
#
# This cell demonstrates the intended resume workflow:
# rebuild session/trainer, then load.

# %%
# %% [resume-from-checkpoint]
session2, trainer2 = make_session_and_trainer(cfg, device="auto")
trainer2.load(checkpoint_path)
trainer2.inspect_best(plot=True)

# %% [markdown]
# ---
# ## 5) Full wiring smoke tests for other tasks
#
# We keep these tests **short**:
# - build a session
# - run a reward smoke test on the template
# - run a couple of training epochs (deterministic tasks only)
#
# For SSA tasks, the reward can be expensive; we only do reward smoke tests by default.

# %%
# %% [smoke-tracking-training]
cfg_tr = Configurator.preset("fast")
cfg_tr.task.kind = "tracking"
cfg_tr.task.n_inputs = 3  # still used to build template input count
cfg_tr.task.t_f = 50.0
cfg_tr.task.N_t = 120
cfg_tr.task.weights = "transient"
cfg_tr.task.target = "copy_input0"
cfg_tr.task.ic_value = 0.01

cfg_tr.train.max_added_reactions = 4
cfg_tr.train.epochs = 5
cfg_tr.train.render_every = 1
cfg_tr.train.seed = 1

session_tr, trainer_tr = make_session_and_trainer(cfg_tr, device="auto")

print("=== Tracking task wiring ===")
print_task_summary(session_tr.task)
run_smoke_reward(session_tr.task, session_tr.crn_template, label="tracking template")

# Tiny training chunk
trainer_tr.run(epochs=cfg_tr.train.epochs, checkpoint_path=None)
trainer_tr.inspect_best(plot=False)

# %%
# %% [smoke-oscillator-training]
cfg_osc = Configurator.preset("fast")
cfg_osc.task.kind = "oscillator"
cfg_osc.task.n_inputs = 1
cfg_osc.task.t_f = 50.0
cfg_osc.task.N_t = 120
cfg_osc.task.ic_value = 0.01

cfg_osc.train.max_added_reactions = 4
cfg_osc.train.epochs = 5
cfg_osc.train.render_every = 1
cfg_osc.train.seed = 2

session_osc, trainer_osc = make_session_and_trainer(cfg_osc, device="auto")

print("=== Oscillator task wiring ===")
print_task_summary(session_osc.task)
run_smoke_reward(session_osc.task, session_osc.crn_template, label="oscillator template")

# Tiny training chunk
trainer_osc.run(epochs=cfg_osc.train.epochs, checkpoint_path=None)
trainer_osc.inspect_best(plot=False)

# %%
# %% [smoke-dose-response-reward]
cfg_dose = Configurator.preset("fast")
cfg_dose.task.kind = "dose_response"
cfg_dose.task.n_inputs = 1
cfg_dose.task.t_f = 50.0
cfg_dose.task.N_t = 120
cfg_dose.task.ic_value = 0.1
cfg_dose.task.weights = "transient"

# dose-response requires a target_fn; we pass it via make_task by rebuilding session.task manually
session_dose, trainer_dose = make_session_and_trainer(cfg_dose, device="auto")

# Replace session task with a dose_response task (keeps the same wiring objects)
session_dose.task = make_task(
    kind="dose_response",
    species_labels=session_dose.species_labels,
    dose_range=(0.0, 10.0, 10),
    target_fn=hill_function,
    ic=("constant", cfg_dose.task.ic_value),
    weights=cfg_dose.task.weights,
    t_f=cfg_dose.task.t_f,
    n_t=cfg_dose.task.N_t,
)

print("=== Dose-response task wiring (reward smoke only) ===")
print_task_summary(session_dose.task)
run_smoke_reward(session_dose.task, session_dose.crn_template, label="dose template")

# %%
# %% [smoke-ssa-reward]
cfg_ssa = Configurator.preset("fast")
cfg_ssa.task.kind = "ssa_tracking"
cfg_ssa.task.n_inputs = 2
cfg_ssa.task.t_f = 30.0
cfg_ssa.task.N_t = 80
cfg_ssa.task.ic_value = 0.0
cfg_ssa.task.weights = "steady_state"
cfg_ssa.task.target = "copy_input0"

session_ssa, trainer_ssa = make_session_and_trainer(cfg_ssa, device="auto")

print("=== SSA tracking task wiring (reward smoke only) ===")
print_task_summary(session_ssa.task)
run_smoke_reward(session_ssa.task, session_ssa.crn_template, label="ssa tracking template")

# %%
# %% [smoke-ssa-robust-reward]
cfg_ssa_r = Configurator.preset("fast")
cfg_ssa_r.task.kind = "ssa_robust"
cfg_ssa_r.task.n_inputs = 2
cfg_ssa_r.task.t_f = 30.0
cfg_ssa_r.task.N_t = 80
cfg_ssa_r.task.ic_value = 0.0
cfg_ssa_r.task.weights = "steady_state"
cfg_ssa_r.task.target = "copy_input0"

session_ssa_r, trainer_ssa_r = make_session_and_trainer(cfg_ssa_r, device="auto")

print("=== SSA robust task wiring (reward smoke only) ===")
print_task_summary(session_ssa_r.task)
run_smoke_reward(session_ssa_r.task, session_ssa_r.crn_template, label="ssa robust template")

# %% [markdown]
# ---
# ## 6) Appendix: all tunable knobs
# Everything in the config is editable; the preset only sets defaults.

# %%
# %% [config-describe]
cfg.describe()
