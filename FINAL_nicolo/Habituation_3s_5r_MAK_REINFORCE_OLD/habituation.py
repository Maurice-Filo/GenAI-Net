# %% [markdown]
# # Habituation with recovery and speedup (CVODE)
# 
# 

# %%
import os, sys, numpy as np

print("Python:", sys.version.split()[0])
print("CWD:", os.getcwd())


# %% [markdown]
# ## 1) Import RL4CRN helpers
# 

# %%
from typing import Any, Callable, Dict, List, Sequence, Tuple, Union
import numpy as np
from itertools import product

from RL4CRN.utils.input_interface import (
    register_task_kind,
    overrides_get,
    Configurator,
    TaskKindBase,
    TaskSpec,
)

from RL4CRN.utils.default_tasks.HabituationTaskKind import HabituationGapTaskKind # <-- Gap searches for 2nd and 3rd habituation hallmarks

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
species_labels = ['X_1', 'X_2', 'X_3']
crn, species_labels = build_simple_IOCRN(
    species=species_labels,
    production_input_map={"X_1": "u_1"},
    degradation_input_map={},
    dilution_map={"X_1": 0.1, "X_2": 0.1, "X_3": 0.1},  # add dilution to ensure steady state exists
    production_map={"X_2": 0.1},  # add basal production to X_2 nonzero peaks
    output_species="X_3",
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
# ## 4) Define the task
# 

# %%
HabituationGapTaskKind.pretty_help()

# %%
from RL4CRN.utils.input_interface import make_task, print_task_summary
import numpy as np

# Frequencies: 5s, 10s, 15s periods.
# Keep a fixed ON duration (e.g. 1s) and vary OFF so that (t_on + t_off) = period.
t_on = 1.0
periods = [5.0, 10.0, 15.0]                 # seconds
pulse_shapes = [(t_on, P - t_on) for P in periods]  # [(1,4), (1,9), (1,14)]

# IMPORTANT: Your multifreq loss sorts by period (smaller period = higher freq),
# so pass pulse_shapes in any order; it will be internally ordered.
# Still, it's good practice to provide them explicitly as above.

task = make_task(
    template_crn=crn,
    library_components=library_components,
    kind="habituation_gap",
    species_labels=species_labels,
    params={
        "pulse_shapes": pulse_shapes,   # <-- NEW (list of shapes)
        "n_repeats_pre": 10,
        "n_repeats_post": 10,
        "gap_time": 100.0,
        "n_t": 1000,
        "ic": "from_ss",
        "weights": "transient",
        "max_peak": 10.0,
        "min_peak": 0.1,
        "u_values": [1.0],
        "sensitization": False,

        # Multifreq-specific knobs (optional)
        "freq_weight": 1.0,        # weight of monotonic slope penalty across frequencies
        "gap_weight": 5.0,
        "recovery_tol": 0.05,
        "dishabituate_rho": 1.0,
        "ratio_weights": 1.0,      # or a list
    }
)

print_task_summary(task)

# --- Optional safety checks (recommended) ---
print("Sanity checks:")
print(" - template num_inputs:", crn.num_inputs)
print(" - first u shape:", np.asarray(task.u_list[0]).shape)
print(" - first u length:", len(task.u_list[0]))
assert len(task.u_list[0]) == crn.num_inputs, "Input dimension mismatch: u has wrong length!"

print("Pulse shapes used (t_on, t_off):", pulse_shapes)
print("Periods:", [a + b for a, b in pulse_shapes], " (smaller period = higher frequency)")


# %% [markdown]
# ## 5) Training configuration
# 

# %%
# ---- Train config ----
cfg.train.max_added_reactions = 5
cfg.train.epochs = 31
cfg.train.render_every = 5
cfg.train.seed = 0

# %% [markdown]
# Rendering options

# %%
cfg.render.n_best = 100
cfg.render.disregarded_percentage = 0.9
cfg.render.mode = {  # Mode of the experiment
    'style': 'logger', 
    'task': 'sensitization_gap', 
    'format': 'image',
    'topology': True
}

# %% [markdown]
# ## 7) Create session + trainer
# 

# %%
import os
from datetime import datetime
from pytorch_lightning.loggers import CometLogger

task_name = "Habituation_h3_Task"
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
from RL4CRN.utils.input_interface import make_session_and_trainer
trainer = make_session_and_trainer(cfg, task, logger=logger)

# %% [markdown]
# ## 8) Train and save checkpoints
# 

# %%
# checkpoint_path = "habituation_task_chkpt.pkl"
checkpoint_path = f"{task_name}.pkl"
trainer.run(epochs=cfg.train.epochs, checkpoint_path=checkpoint_path)

trainer.save(checkpoint_path)


