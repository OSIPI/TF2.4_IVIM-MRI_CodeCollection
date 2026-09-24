"""
Script that plots a correlation matrix based on the in-vivo results of a single slice.
"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
from scipy.stats import pearsonr
import nibabel as nib
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize


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
mask_gray = nib.load(r'C:\TF_IVIM_OSIPI/TF2.4_IVIM-MRI_CodeCollection/download/Data/Brain_mask_gray_matter.nii.gz').get_fdata().astype(bool)
mask_white = nib.load(r'C:\TF_IVIM_OSIPI/TF2.4_IVIM-MRI_CodeCollection/download/Data/Brain_mask_white_matter.nii.gz').get_fdata().astype(bool)

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
excluded_algorithms = [
    'TCML_TechnionIIT_lsqlm',          # ID 1
    'TCML_TechnionIIT_lsq_sls_lm'      # ID 3
]
mapping = {algo: i+1 for i, algo in enumerate(all_algorithms)}  # IDs 1..26
analysis_algorithms = [
    alg for alg in all_algorithms
    if alg not in excluded_algorithms
]
# Define new algorithm categories by ID
algorithm_categories = {
    'Nonlinear LS': [2, 4, 5, 6, 7, 8, 9],
    'Variable Projection': [10, 11],
    'Linear LS': [12],
    'Segmented Linear LS': [13, 14],

    'Bayesian': [23, 24],
    'Neural network': [25, 26]
}
# ========================================
# 2. Load datasets
# ========================================
datasets = {}
slice_nr = 12

anatomy = 'Brain'

# read csv
csv_noharm = os.path.join(
    base_dir,
    f'no_harmonization_{anatomy}_slice{slice_nr}.csv'
)

csv_harm = os.path.join(
    base_dir,
    f'bounds_and_initialguess_harmonized_{anatomy}_slice{slice_nr}.csv'
)

datasets_noharm = pd.read_csv(csv_noharm)
datasets_harm = pd.read_csv(csv_harm)

def build_corr_matrix(dataset_algospecific, tissue_mask, param):

    corr_matrix = pd.DataFrame(
        index=analysis_algorithms,
        columns=analysis_algorithms,
        dtype=float
    )

    for alg1 in analysis_algorithms:

        if dataset_algospecific[alg1] is None:
            continue

        p1 = np.array(
            dataset_algospecific[alg1][param]
        ).flatten()[tissue_mask.flatten()]

        for alg2 in analysis_algorithms:

            if dataset_algospecific[alg2] is None:
                continue

            p2 = np.array(
                dataset_algospecific[alg2][param]
            ).flatten()[tissue_mask.flatten()]

            valid_mask = (
                ~np.isnan(p1) &
                ~np.isnan(p2)
            )

            if np.any(valid_mask):

                x = p1[valid_mask]
                y = p2[valid_mask]

                if param == "f_fit":
                    x = np.clip(x, 0, 1)
                    y = np.clip(y, 0, 1)

                r, _ = pearsonr(x, y)

                corr_matrix.loc[alg1, alg2] = r

            else:
                corr_matrix.loc[alg1, alg2] = np.nan

    return corr_matrix

# ========================================
# 3. Helper: draw category bands
# ========================================
def add_category_bands_heatmap(ax, algorithm_categories, orientation='x'):
    cmap = plt.get_cmap('tab20')
    num_categories = len(algorithm_categories)
    start_pos = 0
    n_algos = len(analysis_algorithms)
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
def plot_correlation_matrices_tissue(tissue_mask, tissue_name, slice_idx):

    """
    Lower triangle = harmonized correlations
    Upper triangle = no-harmonization - harmonization
    """

    parameters = ['D_fit', 'f_fit', 'Dp_fit']
    n_algos = len(analysis_algorithms)
    algorithms_ordered = analysis_algorithms

    # ==========================================================
    # BUILD ALGORITHM-SPECIFIC DATASETS
    # ==========================================================
    dataset_noharm = {}
    dataset_harm = {}

    for alg in analysis_algorithms:

        df_no = datasets_noharm[
            datasets_noharm['ivim_algorithm'] == alg
        ]

        df_h = datasets_harm[
            datasets_harm['ivim_algorithm'] == alg
        ]

        dataset_noharm[alg] = None if df_no.empty else df_no
        dataset_harm[alg] = None if df_h.empty else df_h

    # ==========================================================
    # GLOBAL DIFFERENCE SCALE
    # ==========================================================
    all_diffs = []

    for param in parameters:

        corr_no = build_corr_matrix(
            dataset_noharm,
            tissue_mask,
            param
        )

        corr_harm = build_corr_matrix(
            dataset_harm,
            tissue_mask,
            param
        )

        diff = corr_no - corr_harm

        all_diffs.append(diff.values)

    all_diffs = np.concatenate(
        [d.flatten() for d in all_diffs]
    )

    global_vmax_diff = np.nanmax(
        np.abs(all_diffs)
    )

    # ==========================================================
    # FIGURE
    # ==========================================================
    fig, axes = plt.subplots(1, 3, figsize=(21, 9))

    for ax, param in zip(axes, parameters):

        corr_no = build_corr_matrix(
            dataset_noharm,
            tissue_mask,
            param
        )

        corr_harm = build_corr_matrix(
            dataset_harm,
            tissue_mask,
            param
        )

        diff = corr_no - corr_harm

        n = len(corr_harm)

        # ======================================================
        # COMBINED MATRIX
        # ======================================================
        combined = np.full((n, n), np.nan)

        for i in range(n):
            for j in range(n):

                if i > j:
                    # lower triangle = harmonized correlations
                    combined[i, j] = corr_harm.iloc[i, j]

                elif i < j:
                    # upper triangle = difference
                    combined[i, j] = diff.iloc[i, j]

        np.fill_diagonal(combined, np.nan)

        # ======================================================
        # MASKS
        # ======================================================
        mask_lower = np.triu(
            np.ones_like(combined, dtype=bool)
        )

        mask_upper = np.tril(
            np.ones_like(combined, dtype=bool)
        )

        # ======================================================
        # LOWER TRIANGLE (HARMONIZED CORRELATIONS)
        # ======================================================
        sns.heatmap(
            combined,
            mask=mask_lower,
            cmap='viridis',
            vmin=0,
            vmax=1,
            square=True,
            cbar=False,
            ax=ax
        )

        # ======================================================
        # UPPER TRIANGLE (DIFFERENCE)
        # ======================================================
        sns.heatmap(
            combined,
            mask=mask_upper,
            cmap='bwr',
            vmin=-global_vmax_diff,
            vmax=global_vmax_diff,
            square=True,
            cbar=False,
            ax=ax
        )

        # ======================================================
        # AXES
        # ======================================================
        ax.set_xlabel("Algorithm ID", fontsize=18)
        ax.set_ylabel("Algorithm ID", fontsize=18)

        ax.set_xticks(np.arange(n_algos) + 0.5)
        ax.set_yticks(np.arange(n_algos) + 0.5)

        ax.set_xticklabels(
            [mapping[a] for a in algorithms_ordered],
            rotation=90,
            fontsize=18
        )

        ax.set_yticklabels(
            [mapping[a] for a in algorithms_ordered],
            rotation=0,
            fontsize=18
        )

        # ======================================================
        # TITLE
        # ======================================================
        title_map = {
            'D_fit': 'D',
            'f_fit': 'f',
            'Dp_fit': 'D*'
        }

        ax.set_title(
            title_map[param],
            fontsize=20,
            weight='bold'
        )

        # ======================================================
        # CATEGORY BANDS
        # ======================================================
        add_category_bands_heatmap(
            ax,
            algorithm_categories,
            orientation='x'
        )

        add_category_bands_heatmap(
            ax,
            algorithm_categories,
            orientation='y'
        )

    # ==========================================================
    # COLORBAR 1: CORRELATIONS
    # ==========================================================
    sm_corr = ScalarMappable(
        cmap='viridis',
        norm=Normalize(vmin=0, vmax=1)
    )
    sm_corr.set_array([])

    cbar_ax1 = fig.add_axes([0.20, 0.18, 0.60, 0.025])
    cbar1 = fig.colorbar(
        sm_corr,
        cax=cbar_ax1,
        orientation='horizontal'
    )

    cbar1.set_label(
        'Pearson correlation coefficient',
        fontsize=16
    )

    cbar1.ax.tick_params(labelsize=16)

    # ==========================================================
    # COLORBAR 2: DIFFERENCE
    # ==========================================================
    sm_diff = ScalarMappable(
        cmap='bwr',
        norm=Normalize(
            vmin=-global_vmax_diff,
            vmax=global_vmax_diff
        )
    )
    sm_diff.set_array([])

    cbar_ax2 = fig.add_axes([0.20, 0.09, 0.60, 0.025])

    cbar2 = fig.colorbar(
        sm_diff,
        cax=cbar_ax2,
        orientation='horizontal'
    )

    cbar2.set_label(
        r'$\Delta$ Pearson correlation coefficient '
        r'(No harmonization − Harmonized)',
        fontsize=16
    )

    cbar2.ax.tick_params(labelsize=16)

    # ==========================================================
    # LAYOUT
    # ==========================================================
    fig.subplots_adjust(
        left=0.05,
        right=0.98,
        top=0.88,
        bottom=0.30,
        wspace=0.05
    )

    # ==========================================================
    # SAVE
    # ==========================================================
    save_path = (
        rf"C:\TF_IVIM_OSIPI\TF2.4_IVIM-MRI_CodeCollection"
        rf"\codecharacterization\invivo\results"
        rf"\CorrelationMatrixDifference_{anatomy}_{tissue_name}_slice{slice_idx}.png"
    )

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches='tight'
    )

    plt.show()

    print(f"Saved: {save_path}")


# ========================================
# 5. Generate for all parameters & tissues
# ========================================
parameters = ['D', 'f', 'D*']
tissues = {'Gray Matter': 'gray', 'White Matter': 'white'}

for tissue_name, mask_key in tissues.items():
    tissue_mask = masks[mask_key].astype(bool)
    tissue_mask = tissue_mask[:,:,slice_nr]
    plot_correlation_matrices_tissue(tissue_mask, tissue_name, slice_nr)

