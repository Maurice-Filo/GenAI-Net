# %% [markdown]
# # RL4CRN tutorial notebook: RPA (CVODE)
# 
# Refer to the Logic Circuits tutorial for more information about the overall pipeline.
# 

# %%
import os, sys, numpy as np

print("Python:", sys.version.split()[0])
print("CWD:", os.getcwd())


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
from RL4CRN.utils.default_tasks.TrackingTaskKind import TrackingTaskKind


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
species_labels = ['X_1', 'X_2', 'X_3']
crn, species_labels = build_simple_IOCRN(
    species=species_labels,
    production_input_map={"X_1": "u_1"},
    degradation_input_map={"X_3": "u_2"},
    dilution_map={},
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
# ## 4) Define the task: RPA

# %%
TrackingTaskKind.pretty_help()

# %%
task = make_task(
    template_crn=crn,
    library_components=library_components,
    kind="tracking",
    species_labels=species_labels,
    params={
        "t_f": 100,
        "n_t": 1000,
        "ic": ("constant", 0.01),
        "weights": "transient",
        "u_values": [0.5, 1.0, 1.5], # the product (per input) of all the combinations of these values will be used 
        "target" : lambda u_1: u_1,  # target is a callable of the inputs (resolved via input_idx_dict)
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
cfg.train.max_added_reactions = 5
cfg.train.epochs = 101
cfg.train.render_every = 5
cfg.train.seed = 0

# %%
cfg.render.n_best = 50
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

# %%
import os
from datetime import datetime
from pytorch_lightning.loggers import CometLogger

task_name = "RPA_Task"
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
checkpoint_path = "RPA_task_chkpt.pkl"
trainer.run(epochs=cfg.train.epochs, checkpoint_path=checkpoint_path)

# %% [markdown]
# ## 9) Inspect the best CRN
# 

# %%
trainer.inspect_best(plot=True)

best = trainer.best_crn()
print("Hall of Fame size:", len(trainer.s.mult_env.hall_of_fame))
if best is not None:
    print("Best loss:", best.last_task_info.get("reward", None))


# %% [markdown]
# ## 10) Sample and re-simulate

# %%
trainer.sample(10, 10, ic=("constant", 1.0))

# %% [markdown]
# We can now inspect newly sampled I/O CRNs.

# %%
import matplotlib.pyplot as plt

index = 0
crn_s = trainer.get_sampled_crns()[index]
print(crn_s)
print("reward:", crn_s.last_task_info.get("reward", None))

# Plotters depend on your IOCRN implementation
crn_s.plot_transient_response(); plt.show()


# %% [markdown]
# Save again our results.

# %%
trainer.save(checkpoint_path)

# %% [markdown]
# ## 11) Loading a saved Session/Trainer from a checkpoint
# 

# %%
from RL4CRN.utils.input_interface import load_session_and_trainer

trainer_loaded = load_session_and_trainer(checkpoint_path, device="cuda")
trainer_loaded.inspect_best()

# %% [markdown]
# ## 12) Re-simulate Hall-of-Fame CRNs under new conditions

# %%
hof_crns = [item.state for item in trainer.s.mult_env.hall_of_fame]

trainer.s.crn_template

crns_new = trainer.resimulate(
    hof_crns,
    ic=("constant", 0.4), 
    u_spec=("grid", [0.0, 1.0]),
)

trainer.inspect(crns_new[0])
crns_new[0].plot_transient_response(); plt.show()



