import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ========================================
# 1. Define algorithm categories (NEW VERSION)
# ========================================

all_algorithms = [
    'TCML_TechnionIIT_lsqlm', 'TCML_TechnionIIT_lsqtrf', 'TCML_TechnionIIT_lsq_sls_lm',
    'TCML_TechnionIIT_lsqBOBYQA', 'TCML_TechnionIIT_lsq_sls_trf', 'TCML_TechnionIIT_lsq_sls_BOBYQA',
    'ASD_MemorialSloanKettering_QAMPER_IVIM', "IAR_LU_biexp",
    "OGC_AmsterdamUMC_biexp", 'IAR_LU_modified_mix', 'IAR_LU_modified_topopro',
    'ETP_SRI_LinearFitting', "TF_reference_IVIMfit", 'PvH_KB_NKI_IVIMfit',
    'TCML_TechnionIIT_SLS', "IAR_LU_segmented_2step", "IAR_LU_segmented_3step",
    "IAR_LU_subtracted", 'OGC_AmsterdamUMC_biexp_segmented', 'PV_MUMC_biexp',
    "OJ_GU_seg", 'OJ_GU_segMATLAB',
    'OGC_AmsterdamUMC_Bayesian_biexp', 'OJ_GU_bayesMATLAB',
    'IVIM_NEToptim', 'Super_IVIM_DC'
]

mapping = {algo: i + 1 for i, algo in enumerate(all_algorithms)}  # IDs 1..26

algorithm_categories = {
    'Nonlinear LS': [1, 2, 3, 4, 5, 6, 7, 8, 9],
    'Variable Projection': [10, 11],
    'Linear LS': [12],
    'Segmented Linear LS': [13, 14],
    'Segmented Nonlinear LS': [15, 16, 17, 18, 19, 20, 21, 22],
    'Bayesian': [23, 24],
    'Neural network': [25, 26]
}

# ========================================
# 2. Load and preprocess data
# ========================================

file_path = r'C:\TF_IVIM_OSIPI\TF2.4_IVIM-MRI_CodeCollection\test_output_SNR20.csv'
df = pd.read_csv(file_path)

for col in ['D_fitted', 'f_fitted', 'Dp_fitted']:
    df[col] = (
        df[col]
        .astype(str)
        .str.replace('[\[\]]', '', regex=True)
        .replace('nan', np.nan)
        .astype(float)
    )

numeric_columns = ['f', 'Dp', 'D', 'f_fitted', 'Dp_fitted', 'D_fitted']
for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Replace algorithm names with integer IDs
df['Algorithm'] = df['Algorithm'].map(mapping)
df = df.dropna(subset=['Algorithm'])
df['Algorithm'] = df['Algorithm'].astype(int)

# Add VoxelID if missing
if 'VoxelID' not in df.columns:
    df['VoxelID'] = df.groupby(['Region', 'SNR', 'Algorithm']).cumcount()

# ========================================
# 3. Helper: Add category bands
# ========================================

def add_category_bands_heatmap(ax, algorithm_categories, orientation='x'):
    """
    Draw colored category bands just outside the heatmap —
    below (for x) or left (for y), aligned with cells.
    """
    import matplotlib.patches as patches
    cmap = plt.get_cmap('tab20')
    num_categories = len(algorithm_categories)
    n_algos = sum(len(v) for v in algorithm_categories.values())

    start_pos = 0
    for i, (cat, algos) in enumerate(algorithm_categories.items()):
        color = cmap(i / num_categories)
        band_thickness = 0.08 * n_algos  # scales with matrix size

        if orientation == 'x':
            # BELOW the matrix
            rect = patches.Rectangle(
                (start_pos, n_algos),      # x, y (just below last row)
                len(algos),                      # width
                band_thickness,                  # height
                facecolor=color,
                alpha=0.3,
                transform=ax.transData,
                clip_on=False,
                zorder=2,
            )
        else:
            # LEFT of the matrix
            rect = patches.Rectangle(
                (-band_thickness, start_pos),  # x, y
                band_thickness,                      # width
                len(algos),                          # height
                facecolor=color,
                alpha=0.3,
                transform=ax.transData,
                clip_on=False,
                zorder=2,
            )

        ax.add_patch(rect)
        start_pos += len(algos)

    # Separator lines between categories
    start_pos = 0
    for cat, algos in list(algorithm_categories.items())[:-1]:
        start_pos += len(algos)
        if orientation == 'x':
            ax.axvline(start_pos, color='gray', linestyle='--', linewidth=1, zorder=3)
        else:
            ax.axhline(start_pos, color='gray', linestyle='--', linewidth=1, zorder=3)
# ========================================
# 4. Plot correlation matrix
# ========================================

# ========================================
# 4+5. Plot all parameters in one figure
# ========================================

def plot_all_correlation_matrices(df, region, snr, algorithm_categories, output_path=None):
    """
    Plot D, f, and Dp correlation matrices in one row with consistent size and a shared colorbar.
    """
    params = ['D', 'f', 'Dp']
    algorithms_ordered = [a for ids in algorithm_categories.values() for a in ids]
    n_algos = len(algorithms_ordered)

    fig, axes = plt.subplots(1, 3, figsize=(21, 9))
    vmin, vmax = 0, 1
    last_heatmap = None

    for ax, param in zip(axes, params):
        df_filtered = df[(df['Region'] == region) & (df['SNR'] == snr)]
        pivot_df = df_filtered.pivot_table(
            index='VoxelID',
            columns='Algorithm',
            values=f'{param}_fitted'
        )

        if pivot_df.shape[1] < 2:
            print(f"Not enough data for {param}, Region: {region}, SNR: {snr}")
            continue

        corr_matrix = pivot_df.corr().reindex(index=algorithms_ordered, columns=algorithms_ordered)

        # Plot heatmap (no colorbar per subplot)
        hm = sns.heatmap(
            corr_matrix,
            annot=False,
            cmap='viridis',
            square=True,
            vmin=vmin, vmax=vmax,
            cbar=False,
            ax=ax
        )
        last_heatmap = hm

        # Axis labels
        ax.set_xlabel("Algorithm ID", fontsize=18)
        ax.set_ylabel("Algorithm ID", fontsize=18)
        ax.set_xticks(np.arange(n_algos) + 0.5)
        ax.set_yticks(np.arange(n_algos) + 0.5)
        ax.set_xticklabels(algorithms_ordered, rotation=90, fontsize=18)
        ax.set_yticklabels(algorithms_ordered, rotation=0, fontsize=18)

        # Subplot title
        ax.set_title(param if param != 'Dp' else 'D*', fontsize=16, weight='bold')

        # Category bands
        add_category_bands_heatmap(ax, algorithm_categories, orientation='x')
        add_category_bands_heatmap(ax, algorithm_categories, orientation='y')

    # Shared colorbar below all subplots
    cbar_ax = fig.add_axes([0.25, 0.10, 0.5, 0.02])  # [left, bottom, width, height]
    cbar = fig.colorbar(last_heatmap.collections[0], cax=cbar_ax, orientation='horizontal')
    cbar.set_label('Pearson correlation coefficient', fontsize=18)
    cbar.ax.tick_params(labelsize=18)

    # Adjust layout to leave space for colorbar and title
    fig.subplots_adjust(left=0.03, right=1, top=0.85, bottom=0.2, wspace=0)

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()




# ========================================
# Run for one SNR/region
# ========================================

regions = df['Region'].unique()
snrs = sorted(df['SNR'].unique())

for region in regions:
    for snr in snrs:
        print(f"\n=== Region: {region}, SNR: {snr} ===")
        output_path = rf'C:\TF_IVIM_OSIPI\TF2.4_IVIM-MRI_CodeCollection\correlationmatrices_allparams_{region}_SNR{int(snr)}.png'
        plot_all_correlation_matrices(df, region, snr, algorithm_categories, output_path)
        print(f"Saved combined correlation matrix figure for SNR {int(snr)}")
