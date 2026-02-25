"""
Lightweight wrappers around GPU-accelerated SSA runs.

This module provides a convenience function to run stochastic simulations
(via `StochasticSimulationsNew.MultiGPUsupport.SSA`) over many parameter/input
configurations and return a *summarized* pandas DataFrame suitable for quick
analysis and plotting.

Key idea:
    - The backend SSA typically returns **raw per-trajectory samples** (including
    metadata like GPU/thread indices).
    - `quick_measurement_SSA` groups the raw output by time and parameter columns,
    and computes **mean** and **standard deviation** trajectories for selected
    species, yielding a compact table (often with MultiIndex columns:
    `(species, 'mean')`, `(species, 'std')`).

The returned summary is designed to be compatible with downstream plotting
helpers such as `plot_simulation_summary`.
"""

from StochasticSimulationsNew.MultiGPUsupport import SSA, spread_parameter_sets_among_gpus

def quick_measurement_SSA(crn, parameters, parameter_names, t_fin=100, n_trajectories=100, 
                          max_threads=10000, t_step=0.1, t_control_step=-1,
                          species_to_measure=None, max_value=1e6):
    """
    Run GPU-accelerated SSA for multiple parameter settings and return a compact summary.

    This is a thin wrapper around `SSA(...)` that:
    
    1. distributes parameter sets across available GPUs,
    2. runs many SSA trajectories per parameter set, and
    3. aggregates the raw results into mean/std trajectories per (time, inputs).

    Args:
        crn: Parsed CRN object consumed by the SSA backend.
            Expected to expose `crn.species` (iterable of species names).
        parameters (Sequence[tuple | list]): Parameter/input configurations to simulate.
            Commonly used for input combinations, e.g. `[(0, 0), (0, 1), (1, 0), (1, 1)]`.
        parameter_names (Sequence[str]): Names corresponding to each entry in a parameter tuple.
            These become grouping columns in the output (e.g. `['u_1', 'u_2']`).
        t_fin (float): Final simulation time (end of time horizon).
        n_trajectories (int): Number of independent SSA trajectories per parameter configuration.
        max_threads (int): Maximum number of GPU threads used by the backend.
            (Passed through to the SSA implementation if applicable.)
        t_step (float): Sampling timestep for recording trajectories.
        t_control_step (float): Control/update interval if the backend supports time-varying inputs.
            Use `-1` for “no control stepping” (backend-dependent).
        species_to_measure (list[str] | None): Subset of species names to keep and summarize.

            - If None: summarize all species in `crn.species` that appear in the SSA dataframe.
            - If provided: silently ignores species names not present in the SSA output columns.
        max_value (float): Divergence/overflow clamp passed to the SSA backend. If states exceed
            this value, the backend may flag divergence.

    Returns:
        tuple[pandas.DataFrame, bool]:

            - `summary_df`: Aggregated dataframe with one row per unique combination of
              grouping columns (time + parameters), and MultiIndex measurement columns:
              `(species, 'mean')`, `(species, 'std')`.
              The grouping columns include `time` and all non-species, non-metadata columns
              (e.g., `u_1`, `u_2`, ...).
            - `has_diverged`: True if any trajectory/configuration was flagged as diverged
              in the raw SSA output (based on a `has_diverged` column).

    Notes:
        - The function assumes the raw SSA dataframe includes:
            * species columns (matching `crn.species`),
            * grouping columns (e.g., `time` and parameter/input columns),
            * optional metadata columns such as `thread_index`, `iteration_index`, `gpu`,
            * a boolean `has_diverged` column used to detect unstable simulations.
        - The aggregation is performed via:
            `raw_df.groupby(group_cols)[target_species].agg(['mean','std']).reset_index()`.
    """
    
    # 1. Distribute parameters to GPUs
    parameter_sets = spread_parameter_sets_among_gpus(parameters) # (in this case the parameters are input combinations)

    # print(parameter_sets)
    
    # print(f"Starting Simulation: {len(parameters)} configurations, {n_trajectories} trajectories each...")
    
    # 2. Run Raw Simulation
    raw_df = SSA(crn, parameter_sets, parameter_names, t_fin, n_trajectories, t_step, t_control_step, max_value=max_value)

    # print(raw_df.head())
    
    # 3. Intelligent Column Detection
    # We need to distinguish between:
    # - Metadata (thread info) -> DROP
    # - Species (measurements) -> AGGREGATE
    # - Parameters (inputs/time) -> GROUP BY
    
    # Get species names from the CRN object columns in the df
    # (We filter columns that exist in the CRN definitions)
    all_species_names = [s for s in crn.species]
    
    # Determine which species to calculate stats for
    if species_to_measure:
        target_species = [s for s in species_to_measure if s in raw_df.columns]
    else:
        target_species = [s for s in all_species_names if s in raw_df.columns]

    # known metadata columns produced by the backend
    metadata_cols = {'thread_index', 'iteration_index', 'gpu'}
    
    # "Grouping columns" are Time + Any Input Parameters (u_1, kr, etc.)
    # We define them as: All columns that are NOT species and NOT metadata
    group_cols = [c for c in raw_df.columns 
                  if c not in all_species_names 
                  and c not in metadata_cols]
    
    # 4. Aggregation
    # print("Summarizing data...")
    summary_df = raw_df.groupby(group_cols)[target_species].agg(['mean', 'std']).reset_index()
    
    has_diverged = raw_df['has_diverged'].any()
    
    # if has_diverged:
    #     print("Warning: Some simulations have diverged (rejecting CRN).")
    # else:
    #     print("Info: Simulations completed without divergence.")

    # print("Done.")
    return summary_df, has_diverged
