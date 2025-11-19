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
base_dir = r"C:\TF_IVIM_OSIPI\TF2.4_IVIM-MRI_CodeCollection\CodeCharacterization\invivo\results"


def load_mask(anatomy):
    mask_paths = {
        "gray": '../../download/Data/brain_mask_gray_matter.nii.gz',
        "white": '../../download/Data/brain_mask_white_matter.nii.gz'
    }
    masks = {}
    for key, path in mask_paths.items():
        masks[key] = nib.load(path).get_fdata().astype(bool)
    return masks


masks = load_mask('brain')

# Load masks
mask_gray = nib.load(r'C:\TF_IVIM_OSIPI\TF2.4_IVIM-MRI_CodeCollection\download\Data\brain_mask_gray_matter.nii.gz').get_fdata().astype(bool)
mask_white = nib.load(r'C:\TF_IVIM_OSIPI\TF2.4_IVIM-MRI_CodeCollection\download\Data\brain_mask_white_matter.nii.gz').get_fdata().astype(bool)

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

mapping = {algo: i+1 for i, algo in enumerate(all_algorithms)}  # IDs 1..26

# Define new algorithm categories by ID
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
# 3. Helper: draw category bands
# ========================================
def add_category_bands_heatmap(ax, algorithm_categories, orientation='x'):
    cmap = plt.get_cmap('tab20')
    num_categories = len(algorithm_categories)
    start_pos = 0
    n_algos = len(all_algorithms)
    band_thickness = 0.08 * n_algos

    for i, (cat, algos) in enumerate(algorithm_categories.items()):
        end_pos = start_pos + len(algos)
        color = cmap(i / num_categories)

        if orientation == 'x':
            rect = patches.Rectangle(
                (start_pos, n_algos), len(algos), band_thickness,
                facecolor=color, alpha=0.3, transform=ax.transData, clip_on=False
            )
        else:
            rect = patches.Rectangle(
                (-band_thickness, start_pos), band_thickness, len(algos),
                facecolor=color, alpha=0.3, transform=ax.transData, clip_on=False
            )
        ax.add_patch(rect)
        start_pos = end_pos

    # Separator lines
    start_pos = 0
    for cat, algos in list(algorithm_categories.items())[:-1]:
        start_pos += len(algos)
        if orientation == 'x':
            ax.axvline(start_pos, color='gray', linestyle='--', linewidth=1, zorder=2)
        else:
            ax.axhline(start_pos, color='gray', linestyle='--', linewidth=1, zorder=2)

# ========================================
# 4. Compute correlation matrix & plot
# ========================================
def plot_correlation_matrices_tissue(tissue_mask, tissue_name):
    """
    Plot 3 correlation matrices (D, f, D*) side by side for a given tissue.
    """
    parameters = ['D', 'f', 'D*']
    n_algos = len(all_algorithms)
    algorithms_ordered = all_algorithms

    fig, axes = plt.subplots(1, 3, figsize=(21, 9))
    cmap = 'viridis'
    vmin, vmax = 0, 1
    last_heatmap = None

    for ax, param in zip(axes, parameters):
        corr_matrix = pd.DataFrame(index=algorithms_ordered, columns=algorithms_ordered, dtype=float)
        # Compute correlation matrix
        for alg1 in all_algorithms:
            if datasets[alg1] is None:
                continue
            D1_full = np.array(datasets[alg1][param]).flatten()[tissue_mask.flatten()]
            D1_full = np.where(np.isnan(D1_full), np.nan, D1_full)
            for alg2 in all_algorithms:
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

        # Plot heatmap (no colorbar per subplot)
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

        # Axis labels
        ax.set_xlabel("Algorithm ID", fontsize=18)
        ax.set_ylabel("Algorithm ID", fontsize=18)
        ax.set_xticks(np.arange(n_algos) + 0.5)
        ax.set_yticks(np.arange(n_algos) + 0.5)
        ax.set_xticklabels([mapping[a] for a in algorithms_ordered], rotation=90, fontsize=18)
        ax.set_yticklabels([mapping[a] for a in algorithms_ordered], rotation=0, fontsize=18)

        # Subplot title
        ax.set_title(param if param != 'Dp' else 'D*', fontsize=20, weight='bold')

        # Category bands
        add_category_bands_heatmap(ax, algorithm_categories, orientation='x')
        add_category_bands_heatmap(ax, algorithm_categories, orientation='y')

    # Shared colorbar below all subplots
    cbar_ax = fig.add_axes([0.25, 0.10, 0.5, 0.02])  # moved slightly up + taller
    cbar = fig.colorbar(last_heatmap.collections[0], cax=cbar_ax, orientation='horizontal')
    cbar.set_label('Pearson correlation coefficient', fontsize=18)
    cbar.ax.tick_params(labelsize=18)

    # Adjust layout to give more room for colorbar and titles
    fig.subplots_adjust(left=0.03, right=1, top=0.85, bottom=0.2, wspace=0)

    # Save and show
    save_path = rf"C:\TF_IVIM_OSIPI\TF2.4_IVIM-MRI_CodeCollection\CorrelationMatrix_{tissue_name.replace(' ', '')}_3params_clipped.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

    print(f"Saved: {save_path}")


# ========================================
# 5. Generate for all parameters & tissues
# ========================================
parameters = ['D', 'f', 'D*']
tissues = {'Gray Matter': 'gray', 'White Matter': 'white'}

for tissue_name, mask_key in tissues.items():
    tissue_mask = masks[mask_key].astype(bool)
    plot_correlation_matrices_tissue(tissue_mask, tissue_name)

