# helper functions

import pandas as pd
import matplotlib.pyplot as plt
import math

def plot_simulation_summary(df, species_cols=['X_1', 'X_2', 'X_3', 'OUT'], input_cols=['u_1', 'u_2', 'u_3'], mean_only=False):
    """
    Plots the mean and standard deviation of species trajectories 
    for each unique input combination.
    
    Compatible with:
    1. Raw Data (contains 'thread_index', needs aggregation)
    2. Summarized Data (output of quick_measurement_SSA, already has mean/std)
    """
    
    if df.empty:
        print("DataFrame is empty.")
        return

    # --- Helper to safely access columns (Handles 'u_1' vs ('u_1', '')) ---
    def get_col_name(df, name):
        if name in df.columns: return name
        # If columns are MultiIndex, 'u_1' might be ('u_1', '')
        if isinstance(df.columns, pd.MultiIndex):
            if (name, '') in df.columns: return (name, '')
        return None

    # --- 1. Detect Mode & Prepare Inputs ---
    
    # Check if this is already summarized (look for 'mean' in column levels)
    is_summarized = False
    if isinstance(df.columns, pd.MultiIndex):
        # Check if 'mean' is in the second level of columns
        if 'mean' in df.columns.get_level_values(1):
            is_summarized = True

    print(f"Plotting mode: {'Summarized (Pre-calculated)' if is_summarized else 'Raw (Aggregating now)'}")

    # Identify valid input columns present in the dataframe
    valid_input_cols = []
    col_mapping = {} # Maps 'u_1' -> 'u_1' or ('u_1', '')
    
    for col in input_cols:
        actual_col = get_col_name(df, col)
        if actual_col:
            valid_input_cols.append(actual_col)
            col_mapping[col] = actual_col

    # Get unique combinations
    if not valid_input_cols:
        input_combinations = [("All Data",)]
    else:
        input_combinations = df[valid_input_cols].drop_duplicates().values

    n_plots = len(input_combinations)
    if n_plots == 0: return

    # --- 2. Setup Grid ---
    n_cols = math.ceil(math.sqrt(n_plots))
    n_rows = math.ceil(n_plots / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 12), constrained_layout=True)
    axes_flat = axes.flatten() if n_plots > 1 else [axes]

    colors = {'X_1': 'blue', 'X_2': 'orange', 'X_3': 'green', 'OUT': 'red', 'Z_1': 'purple', 'Z_2': 'cyan'}

    # --- 3. Iterate & Plot ---
    for i, combo in enumerate(input_combinations):
        ax = axes_flat[i]
        
        # Filter Data
        if not valid_input_cols:
             subset = df
             title_str = "All Data"
        else:
            mask = pd.Series(True, index=df.index)
            title_parts = []
            for idx, col_path in enumerate(valid_input_cols):
                val = combo[idx]
                mask &= (df[col_path] == val)
                
                # Clean name for title (remove empty tuple part if exists)
                clean_name = col_path[0] if isinstance(col_path, tuple) else col_path
                title_parts.append(f"{clean_name}={val}")
                
            subset = df[mask]
            title_str = ", ".join(title_parts)
        
        if subset.empty:
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center')
            continue
            
        # Prepare Stats Data
        if is_summarized:
            # Data is already aggregated. We just need to set time as index.
            time_col = get_col_name(subset, 'time')
            stats = subset.set_index(time_col).sort_index()
        else:
            # Raw data: Group and Aggregate
            stats = subset.groupby('time')[species_cols].agg(['mean', 'std'])
        
        # Plotting Loop
        for species in species_cols:
            # Check if species exists in stats (Handle MultiIndex columns)
            # We expect stats[species]['mean'] to work
            if species not in stats.columns.levels[0]:
                continue
                
            mean_vals = stats[species]['mean']
            std_vals = stats[species]['std']
            times = stats.index
            
            if species in colors.keys():
                c = colors.get(species, 'black')
            else:
                # pick a random color within matplotlib tab10
                c = plt.get_cmap('tab10')(hash(species) % 10)
            
            ax.plot(times, mean_vals, label=f'{species}', color=c, linewidth=2)
            if not mean_only:
                ax.fill_between(times, mean_vals - std_vals, mean_vals + std_vals, color=c, alpha=0.2)
            
        ax.set_title(title_str, fontsize=10)
        ax.set_xlabel('Time')
        ax.set_ylabel('Count')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize='x-small', loc='upper right')

    # Hide unused subplots
    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].axis('off')
    
    fig.suptitle('Trajectory Summary by Input Combination', fontsize=16)
    # plt.show()
