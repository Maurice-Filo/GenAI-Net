# %% [markdown]
# # RL4CRN app 17: Habituation Hallmarks Custom
# 
# This notebook runs the custom six-hallmark habituation objective implemented in `apps/habituation/hallmarks.py` through the RL4CRN training loop.
# 
# The task kind is registered as `habituation_hallmarks_custom`.
# 

# %%
import os, sys
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp')

repo_root = Path.cwd().resolve()
for candidate in [repo_root, *repo_root.parents]:
    if (candidate / 'RL4CRN').exists() and (candidate / 'apps').exists():
        repo_root = candidate
        break
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

# Select the default checkpoint path for this specific run script
default_checkpoint_path = 'habituation_hallmarks_custom_chkpt_run_MA_4S_8R-7.pkl'

checkpoint_path_input = input(f'Checkpoint path [{default_checkpoint_path}]: ').strip()
checkpoint_path_raw = checkpoint_path_input or default_checkpoint_path

checkpoint_path_candidate = Path(checkpoint_path_raw).expanduser()
if checkpoint_path_candidate.is_absolute():
    checkpoint_path = checkpoint_path_candidate
else:
    checkpoint_candidates = [
        Path.cwd() / checkpoint_path_candidate,
        repo_root / checkpoint_path_candidate,
        repo_root / 'apps' / checkpoint_path_candidate,
    ]
    checkpoint_path = next((p for p in checkpoint_candidates if p.exists()), checkpoint_candidates[0])

resume_from_checkpoint = checkpoint_path.exists()
comet_experiment_key = os.environ.get('COMET_EXPERIMENT_KEY', '').strip()

print('Checkpoint path:', checkpoint_path)
print('Resume from checkpoint:', resume_from_checkpoint)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# %% [markdown]
# ## 1) Imports

# %%
from RL4CRN.utils.input_interface import Configurator, make_task, make_session_and_trainer, print_task_summary
from RL4CRN.utils.default_tasks.HabituationHallmarksTaskKind import HabituationHallmarksCustomTaskKind
from apps.habituation.hallmarks_helpers import render_habituation

HabituationHallmarksCustomTaskKind.pretty_help()


# %% [markdown]
# ## 2) Template IO/CRN

# %%
from RL4CRN.utils.crn_builders import build_simple_IOCRN

cfg = Configurator.preset('paper')
cfg.solver.algorithm = 'CVODE'
cfg.solver.rtol = 1e-9
cfg.solver.atol = 1e-9

species_labels = ['X_1', 'X_2', 'X_3', 'X_4']
crn, species_labels = build_simple_IOCRN(
    species=species_labels,
    production_input_map={'X_1': 'u_1'},
    degradation_input_map={},
    dilution_map={},
    production_map={},
    output_species='X_4',
    solver=cfg.solver,
)

from RL4CRN.iocrns.reactions import MassAction
input_degradation = MassAction(
    reactant_labels=["X_2"],
    product_labels=[],
    input_channels=["u_1"],
    params=[0.1],
    params_controllability=[False],
)
crn.add_reaction(input_degradation)

print('Template CRN built.')
print(crn)


# %% [markdown]
# ## 3) Reaction Library

# %%
from RL4CRN.utils.library_builders import build_MAK_library
from RL4CRN.iocrns.reaction_library import construct_catalytic_michaelis_mentin_library

library_components = build_MAK_library(crn, species_labels, order=2)
library, M, K, masks = library_components

template_reaction_ids = {r.ID for r in crn.reactions}

bad_ids = [
    r.ID
    for r in library.reactions
    if r.reactant_labels == []
    and r.product_labels != []
    and r.ID not in template_reaction_ids
]
library.remove_reactions(bad_ids, remove_by='ID')

M = len(library.reactions)
K = library.get_num_parameters()
masks = {
    'continuous': library.get_parameter_mask(mode='continuous', force=True),
    'discrete': library.get_parameter_mask(mode='discrete', force=True),
    'logit': library.get_logit_mask(force=True),
}
crn.set_library_context(library)
library_components = library, M, K, masks

print('Removed zero-order production reactions:', len(bad_ids))
print('Library built: M=', M, 'K=', K)


# %% [markdown]
# ## 4) Custom Hallmark Task Parameters

# %%
# Reference pulse protocol.
A = 10.0
T = 15.0
Ton = 1.11
n_pulses = 50

# Hallmark 4 and 5 sweeps.
T_values = [15.0, 20.0, 25.0]
A_values = [10.0, 20.0, 30.0]

# Candidate input values for the RL task. The task uses A explicitly below,
# so this mainly keeps the RL4CRN task interface well-defined.
u_values = [A]

# Component weights.
hallmark_weights = {
    'hallmark1': 3.0,
    'hallmark2': 1.0,
    'hallmark3': 2.0,
    'hallmark4': 4.0,
    'hallmark5': 3.0,
    'hallmark6': 3.0,
}

# Per-loss keyword arguments.
h1_kwargs = {
    'tolerance': 0.01,
    'n_min': 6,
    'eps': 1e-12,
    'LARGE_NUMBER': 1e4,
    'n_dec': 4,
}
h2_kwargs = {
    'recovery_tolerance': 0.05,
    'max_gap': 10000.0,
    'search_depth': 16,
}
h3_kwargs = {
    'n_series': 2,
    'recovery_gap_fraction': 0.5,
    'tolerance': 0.01,
}
h4_kwargs = {'tolerance': 0.01}
h5_kwargs = {'tolerance': 0.01}
h6_kwargs = {
    'stricter_habituation_tolerance': 0.005,
    'recovery_tolerance': 0.05,
    'max_gap': 10000.0,
    'search_depth': 16,
}


# %% [markdown]
# ## 5) Build Task

# %%
task = make_task(
    template_crn=crn,
    library_components=library_components,
    kind='habituation_hallmarks_custom',
    species_labels=species_labels,
    params={
        'A': A,
        'T': T,
        'Ton': Ton,
        'n_pulses': n_pulses,
        'T_values': T_values,
        'A_values': A_values,
        'u_values': u_values,
        'ic': ('constant', 1e-3),
        'weights': hallmark_weights,
        'h1_kwargs': h1_kwargs,
        'h2_kwargs': h2_kwargs,
        'h3_kwargs': h3_kwargs,
        'h4_kwargs': h4_kwargs,
        'h5_kwargs': h5_kwargs,
        'h6_kwargs': h6_kwargs,
        'amplification_factors': {
            'hallmark1': 3.0,
            'hallmark2': 3.0,
            'hallmark3': 3.0,
            'hallmark4': 3.0,
            'hallmark5': 3.0,
            'hallmark6': 3.0,
        },
    },
)

print_task_summary(task)
assert len(task.u_list[0]) == crn.num_inputs


# %% [markdown]
# ## 6) Training Configuration

# %%
cfg.train.max_added_reactions = 8
cfg.train.epochs = 301
cfg.train.render_every = 5
cfg.train.seed = 0
cfg.train.hall_of_fame_size = 30
cfg.train.batch_size = 1280

cfg.agent.risk_scheduler = {'risk': 0.9, 'risk_update': 0.0, 'max_risk': 1.0, 'risk_schedule': 1000}
cfg.policy.entropy_weights_per_head = {"structure": 3.0, "continuous": 1.0, "discrete": 0.0, "input_influence": 0.0}
cfg.policy.continuous_distribution = {
    "type": "lognormal_independent",
}

cfg.render.n_best = 5
cfg.render.disregarded_percentage = 0.9
cfg.render.mode = {
    'style': 'logger',
    'task': 'habituation_hallmarks_custom',
    'format': 'figure',
    'figure_prefix': 'Custom Habituation',
    'figsize': (12, 30),
    'topology': True,
}


# %% [markdown]
# ## 7) Optional Comet Logger and Trainer

# %%
from datetime import datetime

logger = None
if os.environ.get('COMET_API_KEY') and os.environ.get('COMET_WORKSPACE'):
    from pytorch_lightning.loggers import CometLogger
    project_name = os.environ.get('COMET_PROJECT_NAME', 'Habituation_hallmarks_custom')
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    if resume_from_checkpoint and not comet_experiment_key:
        comet_experiment_key = input('Existing Comet experiment key: ').strip()
        if not comet_experiment_key:
            raise ValueError('Resuming from a checkpoint requires the existing Comet experiment key.')

    logger = CometLogger(
        api_key=os.environ['COMET_API_KEY'],
        project=project_name,
        workspace=os.environ['COMET_WORKSPACE'],
        **(
            {'experiment_key': comet_experiment_key, 'mode': 'get'}
            if resume_from_checkpoint
            else {'name': f'{project_name}_{timestamp}'}
        ),
    ).experiment
    logger.log_parameters({
        'A': A,
        'T': T,
        'Ton': Ton,
        'n_pulses': n_pulses,
        'T_values': T_values,
        'A_values': A_values,
        **{f'weight_{k}': v for k, v in hallmark_weights.items()},
    })
elif resume_from_checkpoint:
    raise ValueError(
        'Resuming with Comet continuation requires COMET_API_KEY and COMET_WORKSPACE.'
    )

trainer = make_session_and_trainer(cfg, task, logger=logger)


# %% [markdown]
# ## 8) Train

# %%
if resume_from_checkpoint:
    trainer.load(str(checkpoint_path))
    from RL4CRN.utils.hall_of_fame import HallOfFame

    trainer.s.mult_env.hall_of_fame = (
        HallOfFame(max_size=cfg.train.hall_of_fame_size)
        if cfg.train.hall_of_fame_size > 0
        else None
    )

remaining_epochs = max(0, cfg.train.epochs - trainer.state.epoch)
trainer.run(epochs=remaining_epochs, checkpoint_path=str(checkpoint_path))
