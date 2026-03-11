import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def filter_algorithms_by_harmonization(df, harmonization_step):
    # Categorize algorithms
    def categorize_algorithm(df_algo):
        uses_all_bounds = df_algo[['D_use_bounds', 'Dp_use_bounds', 'f_use_bounds']].all().all()
        uses_all_initial_guess = df_algo[['D_use_initial_guess', 'Dp_use_initial_guess', 'f_use_initial_guess']].all().all()
        uses_any_initial_guess = df_algo[['D_use_initial_guess', 'Dp_use_initial_guess', 'f_use_initial_guess']].any().any()
        if uses_all_bounds and uses_all_initial_guess:
            return 3
        elif uses_any_initial_guess:
            return 2
        else:
            return 1

    algo_categories = {}
    for algo in df['Algorithm'].unique():
        df_algo = df[df['Algorithm'] == algo]
        algo_categories[algo] = categorize_algorithm(df_algo)

    if harmonization_step == 'no_harmonization':
        algorithms_filtered = [algo for algo, cat in algo_categories.items()]
    elif harmonization_step == 'initialguess_harmonized':
        algorithms_filtered = [algo for algo, cat in algo_categories.items() if cat in [2, 3]]
    elif harmonization_step == 'bounds_and_initialguess_harmonized':
        algorithms_filtered = [algo for algo, cat in algo_categories.items() if cat == 3]
    else:
        algorithms_filtered = [algo for algo, cat in algo_categories.items()]

    return algorithms_filtered



def add_category_bands_heatmap(ax, algorithm_categories, orientation='x'):
    import matplotlib.patches as patches
    n_algos = sum(len(v) for v in algorithm_categories.values())
    band_thickness = 0.08 * n_algos

    start_pos = 0
    category_color_map = {
        'Nonlinear LS': '#1f77b4',
        'Variable Projection': '#ff7f0e',
        'Linear LS': '#2ca02c',
        'Segmented Linear LS': '#d62728',
        'Segmented Nonlinear LS': '#9467bd',
        'Bayesian': '#8c564b',
        'Neural network': '#e377c2'
    }

    for cat, algos in algorithm_categories.items():
        color = category_color_map.get(cat, '#cccccc')
        if orientation == 'x':
            rect = patches.Rectangle(
                (start_pos, n_algos),
                len(algos),
                band_thickness,
                facecolor=color,
                alpha=0.3,
                transform=ax.transData,
                clip_on=False,
                zorder=2,
            )
        else:
            rect = patches.Rectangle(
                (-band_thickness, start_pos),
                band_thickness,
                len(algos),
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

        # Only keep filtered algorithms
        pivot_df = pivot_df[algorithms_ordered]

        if pivot_df.shape[1] < 2:
            print(f"Not enough data for {param}, Region: {region}, SNR: {snr}")
            continue

        corr_matrix = pivot_df.corr().reindex(index=algorithms_ordered, columns=algorithms_ordered)

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

        ax.set_xlabel("Algorithm ID", fontsize=18)
        ax.set_ylabel("Algorithm ID", fontsize=18)
        ax.set_xticks(np.arange(n_algos) + 0.5)
        ax.set_yticks(np.arange(n_algos) + 0.5)
        ax.set_xticklabels(algorithms_ordered, rotation=90, fontsize=18)
        ax.set_yticklabels(algorithms_ordered, rotation=0, fontsize=18)
        ax.set_title(param if param != 'Dp' else 'D*', fontsize=16, weight='bold')

        add_category_bands_heatmap(ax, algorithm_categories, orientation='x')
        add_category_bands_heatmap(ax, algorithm_categories, orientation='y')

    cbar_ax = fig.add_axes([0.25, 0.10, 0.5, 0.02])
    cbar = fig.colorbar(last_heatmap.collections[0], cax=cbar_ax, orientation='horizontal')
    cbar.set_label('Pearson correlation coefficient', fontsize=18)
    cbar.ax.tick_params(labelsize=18)
    fig.subplots_adjust(left=0.03, right=1, top=0.85, bottom=0.2, wspace=0)

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')


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
region = "Brain"
SNR = 20
harmonized_bounds = True
harmonized_initialguess = True
if harmonized_bounds and harmonized_initialguess:
    harmonization_step = "bounds_and_initialguess_harmonized"
    file_path = '/home/rnga/dkuppens/TF2.4_IVIM-MRI_CodeCollection/test_output_brain_bounds_and_initialguess_harmonized_SNR20.csv'
elif not harmonized_bounds and harmonized_initialguess:
    harmonization_step = "initialguess_harmonized"
    file_path = '/home/rnga/dkuppens/TF2.4_IVIM-MRI_CodeCollection/test_output_brain_initialguess_harmonized_SNR20.csv'
elif not harmonized_bounds and not harmonized_initialguess:
    harmonization_step = "no_harmonization"
    file_path = '/home/rnga/dkuppens/TF2.4_IVIM-MRI_CodeCollection/test_output_brain_no_harmonization_SNR20.csv'
region = region + "_" + harmonization_step
df = pd.read_csv(file_path)
df= df[df["Region"] == region]
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
# Run for one SNR/region
# ========================================

algorithms_filtered = filter_algorithms_by_harmonization(df, harmonization_step)

# Filter algorithm_categories, keeping original membership
algorithm_categories_filtered = {
    cat: [algo for algo in algos if algo in algorithms_filtered]
    for cat, algos in algorithm_categories.items()
}
algorithm_categories_filtered = {cat: algos for cat, algos in algorithm_categories_filtered.items() if algos}

algorithms_ordered = []
for cat, algos in algorithm_categories_filtered.items():
    algorithms_ordered.extend(algos)

output_path = f'/home/rnga/dkuppens/TF2.4_IVIM-MRI_CodeCollection/CodeCharacterization/{harmonization_step}/Brain_{harmonization_step}/correlationmatrices_{region}_SNR{str(SNR)}.png'
plot_all_correlation_matrices(df, region, SNR, algorithm_categories_filtered, output_path)
print(f"Saved combined correlation matrix figure for SNR {int(SNR)}")

