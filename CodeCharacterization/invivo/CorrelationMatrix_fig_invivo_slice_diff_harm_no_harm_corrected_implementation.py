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
mask_gray = nib.load(
    r'C:\TF_IVIM_OSIPI/TF2.4_IVIM-MRI_CodeCollection/download/Data/Brain_mask_gray_matter.nii.gz'
).get_fdata().astype(bool)

mask_white = nib.load(
    r'C:\TF_IVIM_OSIPI/TF2.4_IVIM-MRI_CodeCollection/download/Data/Brain_mask_white_matter.nii.gz'
).get_fdata().astype(bool)

mask_combined = mask_gray | mask_white

masks = {
    'gray': mask_gray,
    'white': mask_white,
    'combined': mask_combined
}


# ========================================
# 1. Define algorithm categories & IDs
# ========================================

algorithm_categories = {
    'Nonlinear LS': [
        'TCML_TechnionIIT_lsqlm',
        'TCML_TechnionIIT_lsqtrf',
        'TCML_TechnionIIT_lsq_sls_lm',
        'TCML_TechnionIIT_lsqBOBYQA',
        'TCML_TechnionIIT_lsq_sls_trf',
        'TCML_TechnionIIT_lsq_sls_BOBYQA',
        'ASD_MemorialSloanKettering_QAMPER_IVIM',
        'IAR_LU_biexp',
        'OGC_AmsterdamUMC_biexp'
    ],

    'Segmented Nonlinear LS': [
        'TCML_TechnionIIT_SLS',
        'IAR_LU_segmented_2step',
        'IAR_LU_segmented_3step',
        'IAR_LU_subtracted',
        'OGC_AmsterdamUMC_biexp_segmented',
        'PV_MUMC_biexp',
        'OJ_GU_seg',
        'OJ_GU_segMATLAB'
    ],

    'Linear LS': [
        'ETP_SRI_LinearFitting'
    ],

    'Segmented Linear LS': [
        'TF_reference_IVIMfit',
        'PvH_KB_NKI_IVIMfit'
    ],

    'Variable Projection': [
        'IAR_LU_modified_mix',
        'IAR_LU_modified_topopro'
    ],

    'Bayesian': [
        'OGC_AmsterdamUMC_Bayesian_biexp',
        'OJ_GU_bayesMATLAB'
    ],

    'Neural network': [
        'IVIM_NEToptim',
        'Super_IVIM_DC'
    ]
}
algorithms_ordered_names = [
    a
    for v in algorithm_categories.values()
    for a in v
]
mapping = {
    algo: i + 1
    for i, algo in enumerate(algorithms_ordered_names)
}
excluded = [1, 3]

analysis_algorithms = [
    alg
    for alg in algorithms_ordered_names
    if mapping[alg] not in excluded
]

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

def build_combined_matrix_tissue(
        dataset_noharm,
        dataset_harm,
        tissue_mask,
        param):

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

    n = len(corr_no)

    diag_corr = np.full(n, np.nan)

    for k, alg in enumerate(analysis_algorithms):

        if dataset_noharm[alg] is None:
            continue

        if dataset_harm[alg] is None:
            continue

        x = np.array(
            dataset_noharm[alg][param]
        ).flatten()[tissue_mask.flatten()]

        y = np.array(
            dataset_harm[alg][param]
        ).flatten()[tissue_mask.flatten()]

        valid = (
            ~np.isnan(x) &
            ~np.isnan(y)
        )

        if np.sum(valid) > 1:

            x_valid = x[valid]
            y_valid = y[valid]

            if param == "f_fit":
                x_valid = np.clip(x_valid, 0, 1)
                y_valid = np.clip(y_valid, 0, 1)

            diag_corr[k] = np.corrcoef(
                x_valid,
                y_valid
            )[0, 1]

    combined = np.zeros((n, n))

    for i in range(n):
        for j in range(n):

            if i >= j:
                combined[i, j] = corr_no.iloc[i, j]
            else:
                combined[i, j] = diff.iloc[i, j]

    for i in range(n):
        combined[i, i] = diag_corr[i]

    return combined, corr_harm, diff, diag_corr

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
    colors = {
        'Nonlinear LS': '#1f77b4',
        'Segmented Nonlinear LS': '#9467bd',
        'Linear LS': '#2ca02c',
        'Segmented Linear LS': '#d62728',
        'Variable Projection': '#ff7f0e',
        'Bayesian': '#8c564b',
        'Neural network': '#e377c2'
    }
    start_pos = 0
    n_algos = len(analysis_algorithms)
    band_thickness = 0.08 * n_algos

    for cat, algos in algorithm_categories.items():

        kept = [
            a for a in algos
            if mapping[a] not in excluded
        ]

        size = len(kept)

        if size == 0:
            continue

        color = colors.get(cat, '#cccccc')

        if orientation == 'x':
            rect = patches.Rectangle(
                (start_pos, n_algos),
                size,
                band_thickness,
                facecolor=color,
                alpha=0.3,
                transform=ax.transData,
                clip_on=False
            )
        else:
            rect = patches.Rectangle(
                (-band_thickness, start_pos),
                band_thickness,
                size,
                facecolor=color,
                alpha=0.3,
                transform=ax.transData,
                clip_on=False
            )

        ax.add_patch(rect)

        start_pos += size

    # Separator lines
    start_pos = 0

    for cat, algos in list(algorithm_categories.items())[:-1]:

        kept = [
            a for a in algos
            if mapping[a] not in excluded
        ]

        start_pos += len(kept)

        if orientation == 'x':
            ax.axvline(
                start_pos,
                color='white',
                linestyle='--',
                linewidth=1.5,
                alpha=0.9,
                zorder=50
            )

        else:
            ax.axhline(
                start_pos,
                color='white',
                linestyle='--',
                linewidth=1.5,
                alpha=0.9,
                zorder=50
            )

# ========================================
# 4. Compute correlation matrix & plot
# ========================================
def plot_correlation_matrices_tissue(tissue_mask, tissue_name, slice_idx):

    """
    Lower triangle = no-harmonized correlations
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
        combined, corr_harm, diff, diag_corr = \
            build_combined_matrix_tissue(
                dataset_noharm,
                dataset_harm,
                tissue_mask,
                param
            )

        n = len(combined)

        mask_top = np.tril(
            np.ones_like(combined, dtype=bool)
        )

        mask_bottom = np.triu(
            np.ones_like(combined, dtype=bool),
            1
        )

        sns.heatmap(
            combined,
            mask=mask_bottom,
            cmap='viridis',
            vmin=0,
            vmax=1,
            square=True,
            cbar=False,
            ax=ax
        )

        sns.heatmap(
            combined,
            mask=mask_top,
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

        ax.set_xlim(0, n)
        ax.set_ylim(n, 0)

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

        for i in range(n):
            ax.add_patch(
                patches.Rectangle(
                    (i, i),
                    1,
                    1,
                    fill=False,
                    edgecolor='black',
                    linewidth=1.5,
                    zorder=100
                )
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
        rf"\CorrelationMatrixDifference_{anatomy}_{tissue_name}_slice{slice_idx}_new.png"
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
tissues = {
    'GrayWhiteMatter': 'combined',
    'GrayMatter': 'gray',
    'WhiteMatter': 'white'
}

for tissue_name, mask_key in tissues.items():

    tissue_mask = masks[mask_key][:, :, slice_nr]

    plot_correlation_matrices_tissue(
        tissue_mask,
        tissue_name,
        slice_nr
    )

