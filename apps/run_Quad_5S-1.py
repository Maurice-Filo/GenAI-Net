# %% [markdown]
# # RL4CRN tutorial notebook: Dose Response (CVODE)
# 
# Refer to the Logic Circuits tutorial for more information about the overall pipeline.
# 

# %%
import os, sys, numpy as np
from pathlib import Path

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
cfg.solver.rtol = 1e-10
cfg.solver.atol = 1e-10

# build template IO/CRN
species_labels = ['X_1', 'X_2', 'X_3', 'X_4', 'X_5']
crn, species_labels = build_simple_IOCRN(
    species=species_labels,
    production_input_map={"X_1": "u_1", "X_2": "u_2", "X_3": "u_3"},
    degradation_input_map={},
    dilution_map={},
    output_species="X_5",
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
        "weights": "transient",
        "u_spec": ("grid", [0.1, 0.4, 0.7, 1.0]),
        "target": lambda u_1, u_2, u_3: (u_2 + np.sqrt(u_2**2 + 4*u_1*u_3)) / (2*u_1),
    }
)

print_task_summary(task)

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
cfg.train.max_added_reactions = 6
cfg.train.epochs = 301
cfg.train.render_every = 5
cfg.train.seed = 0
cfg.train.hall_of_fame_size = 30
cfg.train.batch_size = 1280

cfg.agent.risk_scheduler = {'risk': 0.9, 'risk_update': 0.0, 'max_risk': 1.0, 'risk_schedule': 1000}
cfg.policy.entropy_weights_per_head = {"structure": 2.0, "continuous": 1.0, "discrete": 0.0, "input_influence": 0.0}

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
    name=f"{task_name}_{timestamp}",
)

logger = logger.experiment

# %%
trainer = make_session_and_trainer(cfg, task, logger=logger)

# %% [markdown]
# ## 8) Train and save checkpoints
# 

# %%
checkpoint_path = "Quadratic_task_chkpt_run-1.pkl"
trainer.run(epochs=cfg.train.epochs, checkpoint_path=checkpoint_path)