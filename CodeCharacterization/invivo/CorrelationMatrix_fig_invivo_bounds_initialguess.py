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
base_dir = r"C:\TF_IVIM_OSIPI\TF2.4_IVIM-MRI_CodeCollection\CodeCharacterization\invivo\results\old_data"

def load_mask():
    mask_gray = nib.load(r'C:\TF_IVIM_OSIPI\TF2.4_IVIM-MRI_CodeCollection\download\Data\brain_mask_gray_matter.nii.gz').get_fdata().astype(bool)
    mask_white = nib.load(r'C:\TF_IVIM_OSIPI\TF2.4_IVIM-MRI_CodeCollection\download\Data\brain_mask_white_matter.nii.gz').get_fdata().astype(bool)
    return {'gray': mask_gray, 'white': mask_white}

masks = load_mask()

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

mapping = {algo: i+1 for i, algo in enumerate(all_algorithms)}  # IDs 1..26

algorithm_categories = {
    'Nonlinear LS': [2, 4, 7, 8],
    'Variable Projection': [11],
    'Linear LS': [],
    'Segmented Linear LS': [14],
    'Segmented Nonlinear LS': [15, 16, 17, 21],
    'Bayesian': [23, 24],
    'Neural network': []
}

# Flatten algorithm IDs for filtering
selected_algo_ids = [i for cat_list in algorithm_categories.values() for i in cat_list]
selected_algorithms = [all_algorithms[i-1] for i in selected_algo_ids]

# Filtered categories for plotting
filtered_algorithm_categories = {
    cat: [i for i in algos if i in selected_algo_ids]
    for cat, algos in algorithm_categories.items()
}

# ========================================
# 2. Load datasets
# ========================================
datasets = {}
for algo in all_algorithms:
    csv_path = os.path.join(base_dir, f'IVIM_brain_{algo}.csv')
    if os.path.exists(csv_path):
        datasets[algo] = pd.read_csv(csv_path)
    else:
        datasets[algo] = None
        print(f"Missing CSV for algorithm: {algo}")

# ========================================
# 3. Helper: draw category bands (improved)
# ========================================
def add_category_bands_heatmap(ax, algorithm_categories, orientation='x'):
    cmap = plt.get_cmap('tab20')
    n_algos = len(selected_algorithms)
    band_thickness = 0.08 * n_algos
    start_idx = 0

    # Flattened list of algorithm IDs in order
    algo_id_order = [mapping[a] for a in selected_algorithms]

    for i, (cat, cat_ids) in enumerate(algorithm_categories.items()):
        if not cat_ids:  # skip empty categories
            continue
        # Find positions of this category in the selected algorithm order
        positions = [algo_id_order.index(a) for a in cat_ids]
        min_pos = min(positions)
        max_pos = max(positions) + 1  # +1 for rectangle width

        color = cmap(i / len(algorithm_categories))

        if orientation == 'x':
            rect = patches.Rectangle(
                (min_pos, n_algos), max_pos - min_pos, band_thickness,
                facecolor=color, alpha=0.3, transform=ax.transData, clip_on=False
            )
        else:
            rect = patches.Rectangle(
                (-band_thickness, min_pos), band_thickness, max_pos - min_pos,
                facecolor=color, alpha=0.3, transform=ax.transData, clip_on=False
            )
        ax.add_patch(rect)

        # Draw separator lines
        if orientation == 'x':
            ax.axvline(max_pos, color='gray', linestyle='--', linewidth=1, zorder=2)
        else:
            ax.axhline(max_pos, color='gray', linestyle='--', linewidth=1, zorder=2)

# ========================================
# 4. Compute correlation matrix & plot (filtered)
# ========================================
def plot_correlation_matrices_tissue_filtered(tissue_mask, tissue_name):
    parameters = ['D', 'f', 'D*']
    n_algos = len(selected_algorithms)
    algorithms_ordered = selected_algorithms

    fig, axes = plt.subplots(1, 3, figsize=(15, 9))
    cmap = 'viridis'
    vmin, vmax = 0, 1
    last_heatmap = None

    for ax, param in zip(axes, parameters):
        corr_matrix = pd.DataFrame(index=algorithms_ordered, columns=algorithms_ordered, dtype=float)

        for alg1 in algorithms_ordered:
            if datasets[alg1] is None:
                continue
            D1_full = np.array(datasets[alg1][param]).flatten()[tissue_mask.flatten()]
            D1_full = np.where(np.isnan(D1_full), np.nan, D1_full)
            for alg2 in algorithms_ordered:
                if datasets[alg2] is None:
                    continue
                D2_full = np.array(datasets[alg2][param]).flatten()[tissue_mask.flatten()]
                valid_mask = ~np.isnan(D1_full) & ~np.isnan(D2_full)
                if np.any(valid_mask):
                    if param == 'f':
                        D1_full = np.clip(D1_full, 0, 1)
                        D2_full = np.clip(D2_full, 0, 1)
                    r, _ = pearsonr(D1_full[valid_mask], D2_full[valid_mask])
                    corr_matrix.loc[alg1, alg2] = r
                else:
                    corr_matrix.loc[alg1, alg2] = np.nan

        hm = sns.heatmap(
            corr_matrix.astype(float),
            annot=False,
            cmap=cmap,
            square=True,
            vmin=vmin, vmax=vmax,
            cbar=False,
            ax=ax
        )
        last_heatmap = hm

        ax.set_xlabel("Algorithm ID", fontsize=14)
        ax.set_ylabel("Algorithm ID", fontsize=14)
        ax.set_xticks(np.arange(n_algos) + 0.5)
        ax.set_yticks(np.arange(n_algos) + 0.5)
        ax.set_xticklabels([mapping[a] for a in algorithms_ordered], rotation=90, fontsize=12)
        ax.set_yticklabels([mapping[a] for a in algorithms_ordered], rotation=0, fontsize=12)
        ax.set_title(param if param != 'Dp' else 'D*', fontsize=16, weight='bold')

        add_category_bands_heatmap(ax, filtered_algorithm_categories, orientation='x')
        add_category_bands_heatmap(ax, filtered_algorithm_categories, orientation='y')

    # Shared colorbar
    cbar_ax = fig.add_axes([0.25, 0.10, 0.5, 0.02])
    cbar = fig.colorbar(last_heatmap.collections[0], cax=cbar_ax, orientation='horizontal')
    cbar.set_label('Pearson correlation coefficient', fontsize=14)
    cbar.ax.tick_params(labelsize=12)

    fig.subplots_adjust(left=0.03, right=1, top=0.85, bottom=0.2, wspace=0)

    save_path = rf"C:\TF_IVIM_OSIPI\TF2.4_IVIM-MRI_CodeCollection\codecharacterization\invivo\CorrelationMatrix_{tissue_name.replace(' ', '')}_3params_filtered.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved: {save_path}")

# ========================================
# 5. Generate for all parameters & tissues
# ========================================
tissues = {'Gray Matter': 'gray', 'White Matter': 'white'}

for tissue_name, mask_key in tissues.items():
    tissue_mask = masks[mask_key].astype(bool)
    plot_correlation_matrices_tissue_filtered(tissue_mask, tissue_name)