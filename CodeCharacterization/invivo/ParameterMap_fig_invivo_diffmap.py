"""
Script that plots the parameter maps for a subset of implementation numbers, which can be chosen at the bottom.
The plot consists of a row without harmonization, with harmonization and a difference map.
"""
import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

# ==========================================
# GLOBAL STYLE (CONSISTENT FONT SYSTEM)
# ==========================================
plt.rcParams.update({
    'font.size': 28,
    'axes.titlesize': 32,
    'axes.labelsize': 28,
    'xtick.labelsize': 20,
    'ytick.labelsize': 28,
    'figure.titlesize': 32
})

# ==========================================
# 1. SETUP
# ==========================================
base_dir = r"C:\TF_IVIM_OSIPI\TF2.4_IVIM-MRI_CodeCollection\CodeCharacterization\invivo\results"

parameters = ['D', 'f', 'Dp']
param_file_map = {'D': 'D', 'f': 'f', 'Dp': 'Dp'}

param_labels = {
    'D': 'D (mm²/s)',
    'f': 'f (a.u.)',
    'Dp': 'D* (mm²/s)'
}

row_labels = [
    "No\nharmonization",
    "Bounds + initial guess\nharmonized",
    "Difference\n(H - NH)"
]

# ==========================================
# 2. ALGORITHMS
# ==========================================
algorithm_categories = {
    'Nonlinear LS': [
        'TCML_TechnionIIT_lsqlm', 'TCML_TechnionIIT_lsqtrf', 'TCML_TechnionIIT_lsq_sls_lm',
        'TCML_TechnionIIT_lsqBOBYQA', 'TCML_TechnionIIT_lsq_sls_trf', 'TCML_TechnionIIT_lsq_sls_BOBYQA',
        'ASD_MemorialSloanKettering_QAMPER_IVIM', "IAR_LU_biexp",
        "OGC_AmsterdamUMC_biexp"
    ],
    'Variable Projection': ['IAR_LU_modified_mix', 'IAR_LU_modified_topopro'],
    'Linear LS': ['ETP_SRI_LinearFitting'],
    'Segmented Linear LS ': ["TF_reference_IVIMfit", 'PvH_KB_NKI_IVIMfit'],
    'Segmented Nonlinear LS': [
        'TCML_TechnionIIT_SLS', "IAR_LU_segmented_2step", "IAR_LU_segmented_3step",
        "IAR_LU_subtracted", 'OGC_AmsterdamUMC_biexp_segmented', 'PV_MUMC_biexp',
        "OJ_GU_seg", 'OJ_GU_segMATLAB'
    ],
    'Bayesian': ['OGC_AmsterdamUMC_Bayesian_biexp', 'OJ_GU_bayesMATLAB'],
    'Neural network': ['IVIM_NEToptim', 'Super_IVIM_DC']
}

all_algorithms = [a for cat in algorithm_categories.values() for a in cat]

# ==========================================
# 3. LOAD DATA
# ==========================================
def load_all(mode):
    data = {}
    for algo in all_algorithms:
        data[algo] = {}
        for p in parameters:
            fp = param_file_map[p]

            if mode == "harm":
                fname = f"bounds_and_initialguess_harmonized_{algo}_brain_slice12_fit_{fp}.nii"
            else:
                fname = f"no_harmonization_{algo}_brain_slice12_fit_{fp}.nii"

            path = os.path.join(base_dir, fname)

            if os.path.exists(path):
                data[algo][p] = nib.load(path).get_fdata()
            elif os.path.exists(path + ".gz"):
                data[algo][p] = nib.load(path + ".gz").get_fdata()
            else:
                data[algo][p] = None
    return data

vol_h = load_all("harm")
vol_nh = load_all("noharm")

# ==========================================
# 4. MAIN PLOT
# ==========================================
def plot_3col_grid(vol_h, vol_nh, selected_numbers,
                   slice_index=0, orientation='axial'):

    selected_algos = [all_algorithms[n-1] for n in selected_numbers]

    n_rows = 3
    n_cols_per_param = len(selected_algos)

    fig, axes_all = plt.subplots(
        n_rows,
        n_cols_per_param * len(parameters),
        figsize=(26, 8)
    )

    plt.subplots_adjust(
        wspace=0.25,
        hspace=0.00,
        left=0.18,
        right=0.98,
        top=0.92,
        bottom=0.35
    )

    axes_all = np.array(axes_all).reshape(
        n_rows,
        n_cols_per_param * len(parameters)
    )

    cmap_main = plt.cm.get_cmap('viridis').copy()
    cmap_main.set_bad('black')

    cmap_diff = plt.cm.get_cmap('seismic').copy()
    cmap_diff.set_bad('black')

    color_ranges = {
        'D': (0, 0.003),
        'f': (0, 1),
        'Dp': (0, 0.1)
    }

    def sl(v):
        if orientation == 'axial':
            return np.rot90(v[:, :, slice_index])
        elif orientation == 'coronal':
            return np.rot90(v[:, slice_index, :])
        else:
            return np.rot90(v[slice_index, :, :])

    # ==========================================
    # 5. PLOTTING
    # ==========================================
    for p_idx, param in enumerate(parameters):

        vmin, vmax = color_ranges[param]
        diffs = []

        for algo in selected_algos:
            h = vol_h[algo][param]
            nh = vol_nh[algo][param]
            if h is None or nh is None:
                continue
            diffs.append(sl(h) - sl(nh))

        vmax_diff = np.nanpercentile(
            np.abs(np.concatenate([d.ravel() for d in diffs])),
            99
        )

        for i, algo in enumerate(selected_algos):

            col = i + p_idx * n_cols_per_param

            h = vol_h[algo][param]
            nh = vol_nh[algo][param]

            if h is None or nh is None:
                for r in range(3):
                    axes_all[r, col].axis('off')
                continue

            h_s = sl(h)
            nh_s = sl(nh)
            diff = h_s - nh_s

            axes_all[0, col].imshow(nh_s, cmap=cmap_main, vmin=vmin, vmax=vmax)
            axes_all[1, col].imshow(h_s, cmap=cmap_main, vmin=vmin, vmax=vmax)
            axes_all[2, col].imshow(diff, cmap=cmap_diff,
                                    vmin=-vmax_diff, vmax=vmax_diff)

            for r in range(3):
                axes_all[r, col].axis('off')

            axes_all[2, col].text(
                0.5, -0.25,
                str(selected_numbers[i]),
                transform=axes_all[2, col].transAxes,
                ha='center',
                fontsize=16,
                fontweight='bold'
            )

        # ==========================================
        # 6. COLORBARS (CLEAN + CONSISTENT)
        # ==========================================
        cols = range(p_idx*n_cols_per_param, (p_idx+1)*n_cols_per_param)
        block = [axes_all[0, c].get_position() for c in cols]

        x0 = min(b.x0 for b in block)
        x1 = max(b.x0 + b.width for b in block)
        param_titles = {
            'D': 'D',
            'f': 'f',
            'Dp': 'D*'
        }
        fig.text(
            (x0 + x1) / 2,
            max(b.y1 for b in block) + 0.03,
            param_titles[param],
            fontsize=20,
            weight='bold',
            ha='center'
        )

        # top colorbar
        cbar_ax = fig.add_axes([x0+0.01, 0.25, x1-x0-0.02, 0.014])
        sm = plt.cm.ScalarMappable(
            cmap=cmap_main,
            norm=plt.Normalize(*color_ranges[param])
        )
        cbar1 = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
        cbar1.set_label(param_labels[param], fontsize=20, labelpad=12)

        # bottom colorbar
        cbar_ax2 = fig.add_axes([x0+0.01, 0.1, x1-x0-0.02, 0.014])
        sm2 = plt.cm.ScalarMappable(
            cmap=cmap_diff,
            norm=plt.Normalize(-vmax_diff, vmax_diff)
        )
        cbar2 = fig.colorbar(sm2, cax=cbar_ax2, orientation='horizontal')
        cbar2.set_label(f"Δ{param_labels[param]}", fontsize=20, labelpad=12)

    # ==========================================
    # 7. ROW LABELS (CONSISTENT SIZE)
    # ==========================================
    for r in range(3):
        pos = axes_all[r, 0].get_position()
        fig.text(
            0.04,
            (pos.y0 + pos.y1) / 2,
            row_labels[r],
            va='center',
            ha='left',
            rotation=0,
            fontsize=18,
            fontweight='bold'
        )
    plt.savefig(r'T:\Ben\7.Conferences\ISMRM2026\ParameterMap_differencemap_1_10_18_24_tighter.png')
    plt.show()

# ==========================================
# 8. RUN
# ==========================================
selected_numbers = [2, 9, 18, 24]
plot_3col_grid(vol_h, vol_nh, selected_numbers, slice_index=0)