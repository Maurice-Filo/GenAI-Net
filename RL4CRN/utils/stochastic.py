from StochasticSimulationsNew.MultiGPUsupport import SSA, spread_parameter_sets_among_gpus

def quick_measurement_SSA(crn, parameters, parameter_names, t_fin=100, n_trajectories=100, 
                          max_threads=10000, t_step=0.1, t_control_step=-1,
                          species_to_measure=None, max_value=1e6):
    """
    Runs the SSA simulation and returns a lightweight, summarized DataFrame
    containing only the Mean and Std Dev of the trajectories.
    
    Args:
        crn: The parsed CRN object.
        parameters: List of parameter tuples (e.g. [(0,0), (0,1)]).
        t_fin, n_trajectories, etc.: Simulation settings.
        species_to_measure (list, optional): specific species names to keep. 
                                             If None, keeps all species.
    
    Returns:
        pd.DataFrame: Summarized data. 
                      Columns: [time, u_1, u_2, ...] + MultiIndex [(Species, mean), (Species, std)]
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
