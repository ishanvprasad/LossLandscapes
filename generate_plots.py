import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap

def generate_publication_plots(csv_filename):
    print(f"Loading data from {csv_filename}...")
    df = pd.read_csv(csv_filename)
    
    # 1. Enforce Categorical Sorting
    # Ensures the X-axis (Load) and Y-axis (Time) are plotted in sequential order
    size_order = ["14m", "31m", "70m", "160m", "410m"]
    df['Model_Size'] = pd.Categorical(df['Model_Size'], categories=[s for s in size_order if s in df['Model_Size'].values], ordered=True)
    df = df.sort_values(['Training_Step', 'Model_Size'])
    
    # 2. Pivot the data into 2D matrices (Y=Time, X=Load)
    cka_matrix = df.pivot(index="Training_Step", columns="Model_Size", values="CKA_Similarity")
    barrier_matrix = df.pivot(index="Training_Step", columns="Model_Size", values="Loss_Barrier")
    trace_matrix = df.pivot(index="Training_Step", columns="Model_Size", values="Avg_Trace")
    
    # Invert the Y-axis so time (Step 1000) starts at the bottom and goes up
    cka_matrix = cka_matrix.iloc[::-1]
    barrier_matrix = barrier_matrix.iloc[::-1]
    trace_matrix = trace_matrix.iloc[::-1]

    # 3. Create the Phase Classification Matrix
    # We define empirical thresholds to classify each cell into a Phase (I, III, or IV)
    phase_matrix = pd.DataFrame(index=cka_matrix.index, columns=cka_matrix.columns, dtype=float)
    
    # Define thresholds (you may need to tweak the trace threshold based on your specific results)
    BARRIER_THRESHOLD = 0.5  # Below this, models are considered "connected"
    CKA_THRESHOLD = 0.8      # Above this, representations are highly similar
    # Using the median trace as a simplistic threshold for "flat" vs "sharp"
    TRACE_THRESHOLD = trace_matrix.median().median() if not np.isnan(trace_matrix.median().median()) else 10000

    for col in phase_matrix.columns:
        for row in phase_matrix.index:
            barrier = barrier_matrix.loc[row, col]
            trace = trace_matrix.loc[row, col]
            cka = cka_matrix.loc[row, col]
            
            if pd.isna(barrier) or pd.isna(trace) or pd.isna(cka):
                phase_matrix.loc[row, col] = np.nan
            elif barrier > BARRIER_THRESHOLD:
                # Poorly connected. If trace is high -> Phase I. If low -> Phase III.
                phase_matrix.loc[row, col] = 1 if trace > TRACE_THRESHOLD else 3
            else:
                # Well connected (Phase IV). Split into A and B based on CKA similarity.
                phase_matrix.loc[row, col] = 4.1 if cka < CKA_THRESHOLD else 4.2

    # 4. Set up the Figure
    sns.set_theme(style="white", font_scale=1.2)
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    fig.suptitle("Evolution of Neural Network Loss Landscapes (Time vs. Load)", fontsize=20, fontweight='bold', y=0.95)

    # --- Plot A: Loss Barrier (Mode Connectivity) ---
    sns.heatmap(barrier_matrix, ax=axes[0, 0], cmap="RdBu_r", center=0, 
                annot=True, fmt=".2f", cbar_kws={'label': 'Loss Barrier'})
    axes[0, 0].set_title("Mode Connectivity (Loss Barrier)", fontsize=16, pad=15)
    axes[0, 0].set_ylabel("Training Step (Time)", fontweight='bold')
    axes[0, 0].set_xlabel("Model Size (Load)", fontweight='bold')

    # --- Plot B: Hessian Trace (Local Sharpness) ---
    # We use a logarithmic colormap because Hessian traces span multiple orders of magnitude
    import matplotlib.colors as colors
    sns.heatmap(trace_matrix, ax=axes[0, 1], cmap="magma", 
                norm=colors.SymLogNorm(linthresh=10, linscale=1, vmin=trace_matrix.min().min(), vmax=trace_matrix.max().max()),
                annot=True, fmt=".0f", cbar_kws={'label': 'Log Hessian Trace'})
    axes[0, 1].set_title("Local Sharpness (Avg Hessian Trace)", fontsize=16, pad=15)
    axes[0, 1].set_ylabel("") 
    axes[0, 1].set_xlabel("Model Size (Load)", fontweight='bold')

    # --- Plot C: CKA Similarity ---
    sns.heatmap(cka_matrix, ax=axes[1, 0], cmap="viridis", vmin=0, vmax=1.0, 
                annot=True, fmt=".2f", cbar_kws={'label': 'Linear CKA'})
    axes[1, 0].set_title("Representation Similarity (CKA)", fontsize=16, pad=15)
    axes[1, 0].set_ylabel("Training Step (Time)", fontweight='bold')
    axes[1, 0].set_xlabel("Model Size (Load)", fontweight='bold')

    # --- Plot D: The Final Phase Taxonomy ---
    # Custom colormap for discrete phases
    cmap_phases = ListedColormap(['#d73027', '#fdae61', '#abd9e9', '#4575b4'])
    bounds = [0.5, 1.5, 3.5, 4.15, 4.5] # Bins for 1 (Phase I), 3 (Phase III), 4.1 (Phase IVA), 4.2 (Phase IVB)
    norm = colors.BoundaryNorm(bounds, cmap_phases.N)
    
    sns.heatmap(phase_matrix, ax=axes[1, 1], cmap=cmap_phases, norm=norm,
                cbar_kws={"ticks": [1, 3, 4.1, 4.2]}, linewidths=1, linecolor='white')
    
    # Format the colorbar for the categorical phases
    colorbar = axes[1, 1].collections[0].colorbar
    colorbar.set_ticklabels(['Phase I\n(Sharp, Disconnected)', 
                             'Phase III\n(Flat, Disconnected)', 
                             'Phase IV-A\n(Flat, Connected, Diff Reps)', 
                             'Phase IV-B\n(Flat, Connected, Same Reps)'])
    
    axes[1, 1].set_title("Global Taxonomy Configuration", fontsize=16, pad=15)
    axes[1, 1].set_ylabel("")
    axes[1, 1].set_xlabel("Model Size (Load)", fontweight='bold')

    # Final visual tweaks
    plt.tight_layout(rect=[0, 0.03, 1, 0.92]) # Leave space for the suptitle
    
    output_filename = csv_filename.replace('.csv', '_plot.png')
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Publication plots successfully saved to {output_filename}")
    plt.show()

if __name__ == "__main__":
    # Replace with the actual filename generated by your run_study.py script
    csv_file = "phase_diagram_results_20260218_221429.csv" 
    
    try:
        generate_publication_plots(csv_file)
    except FileNotFoundError:
        print(f"Error: Could not find '{csv_file}'. Please update the filename at the bottom of this script.")