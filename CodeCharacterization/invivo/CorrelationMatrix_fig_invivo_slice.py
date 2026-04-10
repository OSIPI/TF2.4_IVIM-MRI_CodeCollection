import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
from scipy.stats import pearsonr
import nibabel as nib


# ========================================
# 0. Setup paths and masks
# ========================================
base_dir = r"/home/rnga/dkuppens/TF2.4_IVIM-MRI_CodeCollection/CodeCharacterization/invivo/results/"

# Load masks
mask_gray = nib.load('/home/rnga/dkuppens/TF2.4_IVIM-MRI_CodeCollection/download/Data/Brain_mask_gray_matter.nii.gz').get_fdata().astype(bool)
mask_white = nib.load('/home/rnga/dkuppens/TF2.4_IVIM-MRI_CodeCollection/download/Data/Brain_mask_white_matter.nii.gz').get_fdata().astype(bool)

masks = {'gray': mask_gray, 'white': mask_white}


# ========================================
# 1. Define algorithm categories & IDs
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
    'Nonlinear LS':         [1, 2, 3, 4, 5, 6, 7, 8, 9],
    'Variable Projection':  [10, 11],
    'Linear LS':            [12],
    'Segmented Linear LS':  [13, 14],
    'Segmented Nonlinear LS': [15, 16, 17, 18, 19, 20, 21, 22],
    'Bayesian':             [23, 24],
    'Neural network':       [25, 26]
}

def filter_algorithms_by_harmonization(df, harmonization_step):
    """
    Return the list of algorithm *names* that are relevant for the requested
    harmonization step.  Reads D_use_bounds, Dp_use_bounds, f_use_bounds,
    D_use_initial_guess, Dp_use_initial_guess, f_use_initial_guess directly
    from the DataFrame — identical logic to script 1, but keyed by algorithm
    name instead of integer ID.
    """
    def categorize_algorithm(df_algo):
        uses_any_bounds = df_algo[['use_bounds_D', 'use_bounds_Dp', 'use_bounds_f']].any().any()
        uses_any_initial_guess = df_algo[['use_initial_guess_D', 'use_initial_guess_Dp', 'use_initial_guess_f']].any().any()
        if uses_any_bounds and uses_any_initial_guess:
            return 4
        elif uses_any_bounds:
            return 3
        elif uses_any_initial_guess:
            return 2
        else:
            return 1

    algo_categories = {}
    for algo in df['ivim_algorithm'].unique():
        df_algo = df[df['ivim_algorithm'] == algo]
        algo_categories[algo] = categorize_algorithm(df_algo)

    if harmonization_step == 'no_harmonization':
        return list(algo_categories.keys())
    elif harmonization_step == 'initialguess_harmonized':
        return [algo for algo, cat in algo_categories.items() if cat in (2, 4)]
    elif harmonization_step == 'bounds_harmonized':
        return [algo for algo, cat in algo_categories.items() if cat in (3, 4)]
    elif harmonization_step == 'bounds_and_initialguess_harmonized':
        return [algo for algo, cat in algo_categories.items() if cat == 4]
    else:
        return list(algo_categories.keys())


# ========================================
# 2. Load dataset
# ========================================
slice_nr = 12
harmonization_string = 'bounds_harmonized'
anatomy = 'Brain'
filter_algorithms = True   # If True, only include algorithms relevant for the harmonization step

csv_path = os.path.join(base_dir, f'slice12/{harmonization_string}/{harmonization_string}_{anatomy}_slice{slice_nr}.csv')
if os.path.exists(csv_path):
    datasets = pd.read_csv(csv_path)
else:
    raise FileNotFoundError(f"CSV not found: {csv_path}")


# ========================================
# 3. Apply harmonization filter
# ========================================
id_to_name = {v: k for k, v in mapping.items()}

if filter_algorithms:
    algorithms_filtered = filter_algorithms_by_harmonization(datasets, harmonization_string)
else:
    algorithms_filtered = list(all_algorithms)

# Rebuild algorithm_categories keeping only the filtered algorithms,
# in their original category order (mirrors script 1).
algorithm_categories_filtered = {
    cat: [id_to_name[aid] for aid in ids if id_to_name[aid] in algorithms_filtered]
    for cat, ids in algorithm_categories.items()
}
algorithm_categories_filtered = {
    cat: algos
    for cat, algos in algorithm_categories_filtered.items()
    if algos
}

algorithms_ordered = [
    algo
    for algos in algorithm_categories_filtered.values()
    for algo in algos
]

filter_suffix = harmonization_string if filter_algorithms else 'all_algorithms'
print(f"Filter algorithms  : {filter_algorithms}")
print(f"Harmonization step : {harmonization_string}")
print(f"Algorithms included: {len(algorithms_ordered)} / {len(all_algorithms)}")


# ========================================
# 4. Helper: draw category bands on heatmap axes
# ========================================
def add_category_bands_heatmap(ax, algorithm_categories, orientation='x'):
    category_color_map = {
        'Nonlinear LS':           '#1f77b4',
        'Variable Projection':    '#ff7f0e',
        'Linear LS':              '#2ca02c',
        'Segmented Linear LS':    '#d62728',
        'Segmented Nonlinear LS': '#9467bd',
        'Bayesian':               '#8c564b',
        'Neural network':         '#e377c2'
    }
    n_algos = len(algorithms_ordered)
    band_thickness = 0.08 * n_algos

    start_pos = 0
    for cat, algos in algorithm_categories.items():
        color = category_color_map.get(cat, '#cccccc')
        if orientation == 'x':
            rect = patches.Rectangle(
                (start_pos, n_algos), len(algos), band_thickness,
                facecolor=color, alpha=0.3,
                transform=ax.transData, clip_on=False, zorder=2
            )
        else:
            rect = patches.Rectangle(
                (-band_thickness, start_pos), band_thickness, len(algos),
                facecolor=color, alpha=0.3,
                transform=ax.transData, clip_on=False, zorder=2
            )
        ax.add_patch(rect)
        start_pos += len(algos)

    # Dashed separator lines between categories
    start_pos = 0
    for cat, algos in list(algorithm_categories.items())[:-1]:
        start_pos += len(algos)
        if orientation == 'x':
            ax.axvline(start_pos, color='gray', linestyle='--', linewidth=1, zorder=3)
        else:
            ax.axhline(start_pos, color='gray', linestyle='--', linewidth=1, zorder=3)


# ========================================
# 5. Compute correlation matrix & plot
# ========================================
def plot_correlation_matrices_tissue(tissue_mask, tissue_name, slice_idx):
    """
    Plot 3 correlation matrices (D, f, D*) side by side for a given tissue,
    restricted to the algorithms selected by the harmonization filter.
    """
    parameters = ['D_fit', 'f_fit', 'Dp_fit']
    n_algos = len(algorithms_ordered)

    fig, axes = plt.subplots(1, 3, figsize=(21, 9))
    vmin, vmax = 0, 1
    last_heatmap = None

    # Pre-fetch per-algorithm sub-DataFrames (filtered list only)
    dataset_algospecific = {}
    for alg in algorithms_ordered:
        sub = datasets[datasets['ivim_algorithm'] == alg]
        dataset_algospecific[alg] = sub if not sub.empty else None
        if sub.empty:
            print(f"No data for algorithm: {alg}")

    for ax, param in zip(axes, parameters):
        corr_matrix = pd.DataFrame(
            index=algorithms_ordered, columns=algorithms_ordered, dtype=float
        )

        for alg1 in algorithms_ordered:
            if dataset_algospecific[alg1] is None:
                continue
            D1_full = np.array(dataset_algospecific[alg1][param]).flatten()[tissue_mask.flatten()]

            for alg2 in algorithms_ordered:
                if dataset_algospecific[alg2] is None:
                    continue
                D2_full = np.array(dataset_algospecific[alg2][param]).flatten()[tissue_mask.flatten()]

                valid_mask = ~np.isnan(D1_full) & ~np.isnan(D2_full)
                if np.any(valid_mask):
                    v1 = D1_full[valid_mask]
                    v2 = D2_full[valid_mask]
                    if param == 'f_fit':
                        v1 = np.clip(v1, 0, 1)
                        v2 = np.clip(v2, 0, 1)
                    r, _ = pearsonr(v1, v2)
                    corr_matrix.loc[alg1, alg2] = r
                else:
                    corr_matrix.loc[alg1, alg2] = np.nan

        hm = sns.heatmap(
            corr_matrix.astype(float),
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
        ax.set_xticklabels(
            [mapping[a] for a in algorithms_ordered], rotation=90, fontsize=18
        )
        ax.set_yticklabels(
            [mapping[a] for a in algorithms_ordered], rotation=0, fontsize=18
        )

        display_name = 'D*' if param == 'Dp_fit' else param.replace('_fit', '')
        ax.set_title(display_name, fontsize=20, weight='bold')

        add_category_bands_heatmap(ax, algorithm_categories_filtered, orientation='x')
        add_category_bands_heatmap(ax, algorithm_categories_filtered, orientation='y')

    # Shared horizontal colorbar below all subplots
    cbar_ax = fig.add_axes([0.25, 0.10, 0.5, 0.02])
    cbar = fig.colorbar(last_heatmap.collections[0], cax=cbar_ax, orientation='horizontal')
    cbar.set_label('Pearson correlation coefficient', fontsize=18)
    cbar.ax.tick_params(labelsize=18)
    fig.subplots_adjust(left=0.03, right=1, top=0.85, bottom=0.2, wspace=0)

    save_path = (
        f"/home/rnga/dkuppens/TF2.4_IVIM-MRI_CodeCollection/CodeCharacterization/"
        f"invivo/results/CorrelationMatrix_{anatomy}_{tissue_name}_{filter_suffix}_3params_clipped.png"
    )
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved: {save_path}")


# ========================================
# 6. Generate for all tissues
# ========================================
tissues = {'Gray Matter': 'gray', 'White Matter': 'white'}

for tissue_name, mask_key in tissues.items():
    tissue_mask = masks[mask_key].astype(bool)[:, :, slice_nr]
    plot_correlation_matrices_tissue(tissue_mask, tissue_name, slice_nr)