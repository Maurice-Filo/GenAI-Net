# %% [markdown]
# # RL4CRN tutorial notebook: Logic Circuit Inference Task (CVODE)
# 
# This notebook shows an end-to-end **RL4CRN** workflow:
# 
# 1. **Import utilities** and choose a **configuration preset**
# 2. **Build a template IO-CRN** (species, input/output mapping, solver settings)
# 3. **Build a reaction library** (MAK library in this example)
# 4. **Define a task** (here: *Logic Circuit Inference* objective) using the **new TaskKind interface**
# 5. **Configure training**
# 6. **Create a session + trainer** and run training
# 7. **Inspect the best CRN** from the Hall of Fame
# 8. **Sample and re-simulate** under new conditions
# 9. **Save/load** checkpoints
# 

# %%
import os, sys, numpy as np

print("Python:", sys.version.split()[0])
print("CWD:", os.getcwd())


# %% [markdown]
# ## 1) Import RL4CRN helpers
# 
# We use the user-facing interface utilities from `RL4CRN.utils.input_interface`:
# 
# - `Configurator`: provides presets and override helpers
# - `make_task`: builds a `TaskSpec` with reward function and input scenarios
# - `make_session_and_trainer`: wires up environments, interfaces, policy, and agent
# - `print_task_summary`: quick diagnostic summary for the created task
# 

# %%
from RL4CRN.utils.input_interface import (
    Configurator,
    make_task,
    make_session_and_trainer,
    print_task_summary,
)
# import the task class (in this case LogicTaskKind) that you want to use
from RL4CRN.utils.default_tasks.LogicTaskKind import LogicTaskKind

# %% [markdown]
# ## 2) Build a template IO/CRN
# 
# A **template IO-CRN** defines:
# - the **species** in the model
# - how **inputs** enter the system 
# - how constitutive **dilution/productions** (if any) are applied
# - which species is/are treated as the **output**
# - the **ODE solver** to use and its tolerances
# 
# Here we use the convenience builder `build_logic_IOCRN`, which provides an easy way to construct a CRN with digital inputs and a single output node.
# 

# %%
from RL4CRN.utils.crn_builders import build_logic_IOCRN

# choose preset
cfg = Configurator.preset("paper")

# select simulator and set tolerances
cfg.solver.algorithm = "CVODE"
cfg.solver.rtol = 1e-10
cfg.solver.atol = 1e-10

# build template IO/CRN
n_inputs = 4
species_labels = ['X_1', 'X_2', 'X_3', 'X_4', 'X_5']
crn, species_labels = build_logic_IOCRN(
    n_inputs=n_inputs,
    include_dilution=False,
    solver=cfg.solver,
)

print("Template CRN built.")
print(" - num_inputs:", crn.num_inputs)
print(" - num_species:", len(species_labels))
print(" - species:", species_labels)
print(crn)

# %% [markdown]
# ## 3) Build the reaction library (MAK)
# 
# RL4CRN typically proposes reactions from a **library**.  
# This example uses a MAK library of given order (here `order=2`).
# 
# The builder returns a tuple `(library, M, K, masks)` that the training pipeline needs.
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
# ## 4) Define the task: Logic Circuit Inference
# 
# We define a **Logic Circuit Inference** task.
# 
# The system will automatically generates the truth table input scenarios for the chosen number of inputs for the evaluation.
# 
# 

# %%
# explore all available parameters for the LogicCircuit task
LogicTaskKind.pretty_help()

# %%
logic_fn = lambda x: (x[0] and x[1]) or (x[1] and x[2]) or (x[2] and x[3])  # (x1 AND x2) OR (x2 AND x3) OR (x3 AND x4)

task = make_task(
    template_crn=crn,
    library_components=library_components,
    kind="logic",
    species_labels=species_labels,
    params={
        "n_inputs": n_inputs,
        "t_f": 100,
        "n_t": 1000,
        "ic": ("constant", 0.0),
        "weights": "steady_state",
        "logic_fn": logic_fn,
        # optionally "u_spec": ("grid", [0.0, 1.0]) or "u_list": [...]
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
# Note:
#     - `ic=("constant", 0.01)` sets initial concentrations.
# 
# We have specified the target logic function (`logic_fn`) as
# $$
# f_{\text{LOGIC}}(\mathbf{X}) = (X_1 \land X_2) \lor (X_2 \land X_3) \lor (X_3 \land X_4),
# $$
# 
# in this example.

# %% [markdown]
# ## 5) Training configuration
# 
# We tune:
# - `max_added_reactions`: episode length (how many reactions the agent can add)
# - `epochs`: training iterations
# - `render_every`: print progress cadence
# - `seed`: reproducibility
# 

# %%
# ---- Train config ----
cfg.train.max_added_reactions = 6
cfg.train.epochs = 400
cfg.train.render_every = 5
cfg.train.seed = 2
cfg.train.hall_of_fame_size = 30
cfg.agent.risk_scheduler = {'risk': 0.9, 'risk_update': 0.0, 'max_risk': 1.0, 'risk_schedule': 1000}
cfg.agent.sil_settings = {
    'sil_loss_weight': 1.0,
    'sil_use_adaptive_baseline': False,
    'sil_baseline_annealing_rate': 0.95
}
cfg.agent.entropy_scheduler = {'entropy_weight': 0.0,
                                 'topk_entropy_weight': 1.0,
                                 'remainder_entropy_weight': 1.0,
                                 'entropy_update_coefficient': 1,
                                 'entropy_schedule': 1000,
                                 'minimum_entropy_weight': 0.0}

# %% [markdown]
# rendering options:

# %%
cfg.render.n_best = 10
cfg.render.disregarded_percentage = 0.9
cfg.render.mode = {  # Mode of the experiment
    'style': 'logger', 
    'task': 'transients + logic', # we offer several standard rendering modes for specific task types
    'format': 'image',
    'topology': True
}

# %% [markdown]
# ## 6) Inspect full configuration (optional)
# 
# `cfg.describe()` prints a nested configuration dictionary.
# 

# %%
cfg.describe() # print all hyperparameters

# %% [markdown]
# ## 7) Create session + trainer
# 
# This step wires together all the GenAI-Net training loop:
# - parallel environments
# - observer/tensorizer/actuator/stepper interfaces
# - policy + agent
# - the chosen task reward function
# 
# The returned object:
# - `trainer`: runs rollout → reward eval → policy update loops
# 
# 
# **Optional**: Set up a Comet.ml logger:
# 
# you can monitor in real time the progress of GenAI-Net by using Comet.ml. You can set your run up by creating the environmental variables:
# 
# `COMET_API_KEY` and `COMET_WORKSPACE` (usually your API-KEY and username). Otherwise you can skip this step and pass `logger=None` to the trainer.

# %%
import os
from datetime import datetime
from pytorch_lightning.loggers import CometLogger

task_name = "Logic_Circuit_Ablation_No_Entropy"
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

# %% [markdown]
# trainer creation. Trainer object can be checkpointed, reloaded and reused at will.

# %%
trainer = make_session_and_trainer(cfg, task, logger=logger)

# %% [markdown]
# ## 8) Train and save checkpoints
# 
# We run for `cfg.train.epochs` epochs and periodically save a checkpoint.
# 

# %%
checkpoint_path = "run-noEntropy-3.pkl"
trainer.run(epochs=cfg.train.epochs, checkpoint_path=checkpoint_path)