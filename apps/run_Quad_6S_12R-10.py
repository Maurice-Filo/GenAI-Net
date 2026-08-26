# %% [markdown]
# # RL4CRN tutorial notebook: Dose Response (CVODE)
# 
# Refer to the Logic Circuits tutorial for more information about the overall pipeline.
# 

# %%
import os, sys, numpy as np
from pathlib import Path
from itertools import product

os.environ.setdefault("MPLCONFIGDIR", "/tmp")

repo_root = Path.cwd().resolve()
for candidate in [repo_root, *repo_root.parents]:
    if (candidate / "RL4CRN").exists() and (candidate / "apps").exists():
        repo_root = candidate
        break
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))


print("Python:", sys.version.split()[0])
print("CWD:", os.getcwd())
print("Repo root:", repo_root)

# Specify the default checkpoint path for this run
default_checkpoint_path = "Quadratic_task_chkpt_run_Quad_6S_12R-10.pkl"

checkpoint_path_input = input(f"Checkpoint path [{default_checkpoint_path}]: ").strip()
checkpoint_path_raw = checkpoint_path_input or default_checkpoint_path

checkpoint_path_candidate = Path(checkpoint_path_raw).expanduser()
if checkpoint_path_candidate.is_absolute():
    checkpoint_path = checkpoint_path_candidate
else:
    checkpoint_candidates = [
        Path.cwd() / checkpoint_path_candidate,
        repo_root / checkpoint_path_candidate,
        repo_root / "apps" / checkpoint_path_candidate,
    ]
    checkpoint_path = next((p for p in checkpoint_candidates if p.exists()), checkpoint_candidates[0])

resume_from_checkpoint = checkpoint_path.exists()
comet_experiment_key = os.environ.get("COMET_EXPERIMENT_KEY", "").strip()

if resume_from_checkpoint and not comet_experiment_key:
    comet_experiment_key = input("Existing Comet experiment key: ").strip()
    if not comet_experiment_key:
        raise ValueError("Resuming from a checkpoint requires the existing Comet experiment key.")

print("Checkpoint path:", checkpoint_path)
print("Resume from checkpoint:", resume_from_checkpoint)


# %% [markdown]
# ## 1) Import RL4CRN helpers
# 

# %%
from RL4CRN.utils.input_interface import (
    Configurator,
    make_task,
    make_session_and_trainer,
    print_task_summary,
)
from RL4CRN.utils.default_tasks.DoseResponseTaskKind import DoseResponseTaskKind


# %% [markdown]
# ## 2) Build a template IO/CRN
# 

# %%
from RL4CRN.utils.crn_builders import build_simple_IOCRN

# choose preset
cfg = Configurator.preset("paper")

# select simulator and set tolerances
cfg.solver.algorithm = "CVODE"
cfg.solver.rtol = 1e-6
cfg.solver.atol = 1e-6

# build template IO/CRN
species_labels = ['X_1', 'X_2', 'X_3', 'X_4', 'X_5', 'X_6']
crn, species_labels = build_simple_IOCRN(
    species=species_labels,
    production_input_map={"X_1": "u_1", "X_2": "u_2", "X_3": "u_3"},
    degradation_input_map={},
    dilution_map={},
    output_species="X_6",
    solver=cfg.solver,
)

print("Template CRN built.")
print(" - num_inputs:", crn.num_inputs)
print(" - num_species:", len(species_labels))
print(" - species:", species_labels)


# %% [markdown]
# ## 3) Build the reaction library (MAK)
# 

# %%
from RL4CRN.utils.library_builders import build_MAK_library

# library components
library_components = build_MAK_library(crn, species_labels, order=2)

library, M, K, masks = library_components
print("Library built.")
print(" - M (num reactions in library):", M)
print(" - K (num parameters in library):", K)


# %% [markdown]
# ## 4) Define the task: RPA
# 

# %%
from RL4CRN.utils.input_interface import get_task_kind
get_task_kind("dose_response").pretty_help()

# %%
task = make_task(
    template_crn=crn,
    library_components=library_components,
    kind="dose_response",
    species_labels=species_labels,
    params={
        "t_f": 100,
        "n_t": 1000,
        "ic": ("constant", 0.01),
        "weights": "steady_state",
        "u_spec": ("grid", [0.1, 0.4, 0.8, 1.2]),
        "target": lambda u_1, u_2, u_3: (u_2 + np.sqrt(u_2**2 + 4*u_1*u_3)) / (2*u_1),
        "residual": lambda z, u_1, u_2, u_3: u_1 * z**2 - u_2 * z - u_3,
        "residual_weight": 5.0,
        "steady_state_weight": 0.1,
        "size_weight": 1e-4,
        "hidden_cap": 20.0,
        "randomize_u_each_epoch": True,
        "random_u_points": 64,
        "random_u_low": [0.1, 0.1, 0.1],
        "random_u_high": [1.2, 1.2, 1.2],
    }
)

print_task_summary(task)

# --- Override the compute_reward function
def quadratic_residual_reward(state):
    target_fn = task.params["target"]
    residual_fn = task.params["residual"]

    lambda_quad = task.params.get("residual_weight", 5.0)
    lambda_ss = task.params.get("steady_state_weight", 0.1)
    lambda_size = task.params.get("size_weight", 1e-4)
    hidden_cap = task.params.get("hidden_cap", 100.0)

    eps = 1e-8

    x0_list = task.ic.get_ic(state)

    t, x_list, y_list, last_task_info = state.transient_response(
        task.u_list,
        x0_list,
        task.time_horizon,
        LARGE_NUMBER=task.LARGE_NUMBER,
    )

    losses = []
    targets = []

    for (u, x0), x_traj, y_traj in zip(product(task.u_list, x0_list), x_list, y_list):
        u_1, u_2, u_3 = map(float, np.asarray(u).reshape(-1))

        z = float(y_traj[0, -1])

        target = float(target_fn(u_1=u_1, u_2=u_2, u_3=u_3))
        residual = float(residual_fn(z=z, u_1=u_1, u_2=u_2, u_3=u_3))

        targets.append(np.asarray([target], dtype=np.float32))

        target_loss = abs(z - target) / (abs(target) + eps)

        residual_scale = abs(u_1 * z**2) + abs(u_2 * z) + abs(u_3) + eps
        quadratic_loss = (residual / residual_scale) ** 2

        x_final = x_traj[:, -1]
        dx_final = state.rate_function(0.0, x_final, np.asarray(u, dtype=np.float32))
        steady_state_loss = np.mean((dx_final / (1.0 + np.abs(x_final))) ** 2)

        size_loss = np.mean(np.maximum(0.0, np.abs(x_final) - hidden_cap) ** 2)

        loss = (
            target_loss
            + lambda_quad * quadratic_loss
            + lambda_ss * steady_state_loss
            + lambda_size * size_loss
        )

        losses.append(loss)

    performance = float(np.mean(losses))

    state.last_task_info["reward"] = performance
    state.last_task_info["setpoint"] = targets
    state.last_task_info["initial_conditions"] = x0_list
    state.last_task_info["reward type"] = "quadratic_residual_steady_state"

    return performance, state.last_task_info


task.compute_reward = quadratic_residual_reward

# --- Optional safety checks (recommended) ---
print("Sanity checks:")
print(" - template num_inputs:", crn.num_inputs)
print(" - first u shape:", np.asarray(task.u_list[0]).shape)
print(" - first u length:", len(task.u_list[0]))
assert len(task.u_list[0]) == crn.num_inputs, "Input dimension mismatch: u has wrong length!"


# %% [markdown]
# ## 5) Training configuration

# %%
# ---- Train config ----
cfg.train.max_added_reactions = 12
cfg.train.epochs = 601
cfg.train.render_every = 5
cfg.train.seed = 0
cfg.train.hall_of_fame_size = 30
cfg.train.batch_size = 1280

cfg.agent.risk_scheduler = {'risk': 0.9, 'risk_update': 0.0, 'max_risk': 1.0, 'risk_schedule': 1000}
cfg.policy.entropy_weights_per_head = {"structure": 3.0, "continuous": 1.0, "discrete": 0.0, "input_influence": 0.0}

# %%
# ---- rendering ----
cfg.render.n_best = 10
cfg.render.disregarded_percentage = 0.9
cfg.render.mode = {  # Mode of the experiment
    'style': 'logger', 
    'task': 'transients', 
    'format': 'image',
    'topology': True
}

# %% [markdown]
# ## 6) Inspect full configuration (optional)
# 

# %%
cfg.describe()

# %% [markdown]
# ## 7) Create session + trainer
# 
# This step wires together:
# - parallel environments
# - observer/tensorizer/actuator/stepper interfaces
# - policy + agent
# - the chosen task reward function
# 
# The returned object:
# - `trainer`: runs rollout → reward eval → policy update loops
# 

# %%
import os
from datetime import datetime
from pytorch_lightning.loggers import CometLogger

task_name = "DoseResponse_Quadratic_Task"
timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# Expect these in your environment:
#   COMET_API_KEY   (required)
#   COMET_WORKSPACE (required)
api_key = os.environ["COMET_API_KEY"]
workspace = os.environ["COMET_WORKSPACE"]

logger = CometLogger(
    api_key=api_key,
    project=task_name,
    workspace=workspace,
    **(
        {"experiment_key": comet_experiment_key, "mode": "get"}
        if resume_from_checkpoint
        else {"name": f"{task_name}_{timestamp}"}
    ),
)

logger = logger.experiment

# %%
trainer = make_session_and_trainer(cfg, task, logger=logger)

# %% [markdown]
# ## 8) Train and save checkpoints
# 

# %%
if resume_from_checkpoint:
    trainer.load(str(checkpoint_path))
    from RL4CRN.utils.hall_of_fame import HallOfFame

    trainer.s.mult_env.hall_of_fame = (
        HallOfFame(max_size=cfg.train.hall_of_fame_size)
        if cfg.train.hall_of_fame_size > 0
        else None
    )
    trainer.s.task.compute_reward = quadratic_residual_reward

# Added Code
fixed_u_list = [np.asarray(u, dtype=np.float32).reshape(-1) for u in task.u_list]

def sample_random_u_list(epoch):
    n = int(task.params.get("random_u_points", len(fixed_u_list)))

    low = np.asarray(task.params.get("random_u_low", 0.1), dtype=np.float32)
    high = np.asarray(task.params.get("random_u_high", 1.2), dtype=np.float32)

    rng = np.random.default_rng(cfg.train.seed + 1000003 * int(epoch))

    u_array = rng.uniform(
        low=low,
        high=high,
        size=(n, crn.num_inputs),
    ).astype(np.float32)

    return [u_array[i] for i in range(n)]


_original_step_epoch = trainer.step_epoch

def step_epoch_with_random_inputs():
    if task.params.get("randomize_u_each_epoch", False):
        task.u_list = sample_random_u_list(trainer.state.epoch)
        trainer.s.task.u_list = task.u_list

    return _original_step_epoch()


trainer.step_epoch = step_epoch_with_random_inputs


remaining_epochs = max(0, cfg.train.epochs - trainer.state.epoch)
trainer.run(epochs=remaining_epochs, checkpoint_path=str(checkpoint_path))
