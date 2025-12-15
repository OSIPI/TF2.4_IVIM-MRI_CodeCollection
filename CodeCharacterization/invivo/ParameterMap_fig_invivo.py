import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

##Code to plot a single slice of the in-vivo fitting results

# ==========================================
# 1. Setup
# ==========================================
base_dir = r"C:\TF_IVIM_OSIPI\TF2.4_IVIM-MRI_CodeCollection\CodeCharacterization\invivo\results"

parameters = ['D', 'f', 'Dp']
param_file_map = {'D': 'D', 'f': 'f', 'Dp': 'Dp'}

# Algorithm names by category
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

# Flatten all algorithms into a single list for loading
all_algorithms = [algo for cat in algorithm_categories.values() for algo in cat]
n_algos = len(all_algorithms)  # 26

# Map each algorithm to a number for plotting
algorithm_numbers = {algo: i+1 for i, algo in enumerate(all_algorithms)}

ordered_categories = [
    'Nonlinear LS',
    'Variable Projection',
    'Linear LS',
    'Segmented Linear LS ',
    'Segmented Nonlinear LS',
    'Bayesian',
    'Neural network'
]

cmap = plt.get_cmap('tab20')
category_colors = {
    cat: cmap(i / len(ordered_categories))
    for i, cat in enumerate(ordered_categories)
}

# ==========================================
# 2. Load volumes
# ==========================================
def load_all_volumes(base_dir, algorithms, parameters):
    data_dict = {}
    for algo in algorithms:
        data_dict[algo] = {}
        for param in parameters:
            file_param = param_file_map[param]
            nii_path = os.path.join(base_dir, f"{algo}_brain_fit_{file_param}.nii")
            if os.path.exists(nii_path):
                data = nib.load(nii_path).get_fdata()
                data_dict[algo][param] = data
            elif os.path.exists(nii_path + ".gz"):
                data = nib.load(nii_path + ".gz").get_fdata()
                data_dict[algo][param] = data
            else:
                data_dict[algo][param] = None
                print(f"Missing: {nii_path}")
    return data_dict

volumes = load_all_volumes(base_dir, all_algorithms, parameters)

# ==========================================
# 3. Fixed color ranges
# ==========================================
color_ranges = {
    'D': (0, 0.003),
    'f': (0, 1),
    'Dp': (0, 0.1)
}
colorbar_labels = {
    'D': 'D (mm²/s)',
    'f': 'f (a.u.)',
    'Dp': 'D* (mm²/s)'
}

def color_subplot_backgrounds(fig, axes, algorithm_categories, mapping):
    """
    Color subplot backgrounds by algorithm category.
    Draws rectangles behind subplots in normalized figure coordinates.
    """
    # Create mapping: algorithm ID (int) -> category (str)
    algoid_to_category = {}
    for cat, algos in algorithm_categories.items():
        for algo in algos:
            algo_id = mapping.get(algo)
            if algo_id is not None:
                algoid_to_category[algo_id] = cat

    plt.draw()  # compute layout before getting extents

    # Determine horizontal/vertical padding
    m, n = axes.shape
    bbox00 = axes[0, 0].get_window_extent()
    bbox01 = axes[0, 1].get_window_extent() if n > 1 else bbox00
    bbox10 = axes[1, 0].get_window_extent() if m > 1 else bbox00
    pad_h = bbox01.x0 - bbox00.x0 - bbox00.width
    pad_v =  bbox00.y0 - bbox10.y0 - bbox10.height

    # For converting display → figure coordinates
    inv = fig.transFigure.inverted()

    for i in range(m):
        for j in range(n):
            ax = axes[i, j]
            title = ax.get_title()
            if not title:
                continue
            try:
                algo_id = int(title)
            except ValueError:
                continue
            category = algoid_to_category.get(algo_id)
            if category is None:
                continue

            color = category_colors.get(category, (0.8, 0.8, 0.8, 1.0))
            bbox = ax.get_window_extent()

            # Convert pixel coordinates → figure fraction coordinates
            fig_height = fig.get_figheight()
            top_extension = 3 * fig_height
            bottom_extension = 3  # usually keep bottom minimal
            (x0, y0), (x1, y1) = inv.transform([
                (bbox.x0 - 5, bbox.y0 - bottom_extension),
                (bbox.x1 + 5, bbox.y1 + top_extension)
            ])
            width = x1 - x0
            height = y1 - y0

            rect = plt.Rectangle((x0, y0), width, height,
                                 transform=fig.transFigure,
                                 color=color, alpha=0.3,
                                 zorder=-100)
            fig.patches.append(rect)


# ==========================================
# 4. Plot — 3-column figure, 6×5 per parameter, NaNs black
# ==========================================
def plot_3col_grid(volumes, slice_index=None, orientation='axial'):
    n_rows, n_cols_per_param = 6, 5
    fig, axes_all = plt.subplots(n_rows, n_cols_per_param * len(parameters),
                                 figsize=(20, 10))
    plt.subplots_adjust(wspace=0.02, hspace=0.4)
    axes_all = np.array(axes_all).reshape(n_rows, n_cols_per_param * len(parameters))

    for p_idx, param in enumerate(parameters):
        vmin, vmax = color_ranges[param]

        for i in range(n_rows * n_cols_per_param):
            row = i // n_cols_per_param
            col = (i % n_cols_per_param) + p_idx * n_cols_per_param
            ax = axes_all[row, col]

            if i < n_algos:
                algo = all_algorithms[i]
                vol = volumes[algo][param]

                if vol is None:
                    ax.axis('off')
                    ax.set_facecolor("lightgray")
                    ax.text(0.5, 0.5, "Missing", ha='center', va='center', fontsize=8)
                    continue

                if slice_index is None:
                    slice_index = vol.shape[2] // 2

                if orientation == 'axial':
                    img = np.rot90(vol[:, :, slice_index])
                elif orientation == 'coronal':
                    img = np.rot90(vol[:, slice_index, :])
                elif orientation == 'sagittal':
                    img = np.rot90(vol[slice_index, :, :])
                else:
                    raise ValueError("orientation must be 'axial', 'coronal', or 'sagittal'")

                # Mask NaNs and make them black
                masked_img = np.ma.masked_invalid(img)
                cmap = plt.cm.get_cmap('viridis').copy()
                cmap.set_bad(color='black')
                ax.imshow(masked_img, cmap=cmap, vmin=vmin, vmax=vmax)
                ax.axis('off')

                # Algorithm number above each tile
                ax.set_title(f"{algorithm_numbers[algo]}", fontsize=18, pad=1)

                # Leftmost column label
                if (col - p_idx * n_cols_per_param) == 0:
                    ax.set_ylabel(algo, fontsize=7, rotation=0, labelpad=50, va='center')
            else:
                ax.axis('off')


    plt.tight_layout(rect=[0, 0.1, 0.95, 0.95])  # adjust top to leave space
    plt.draw()  # ensure layout is finalized
    n_cols_per_param = 5
    block_spacing = 0.02  # fraction of figure width

    for p_idx in range(1, len(parameters)):  # skip first block
        cols = range(p_idx * n_cols_per_param, (p_idx + 1) * n_cols_per_param)
        for c in cols:
            for r in range(n_rows):
                pos = axes_all[r, c].get_position()
                new_x0 = pos.x0 + block_spacing * p_idx
                axes_all[r, c].set_position([new_x0, pos.y0, pos.width, pos.height])

    for p_idx, param in enumerate(parameters):
        # get the full block of subplots for this parameter
        cols = range(p_idx * n_cols_per_param, (p_idx + 1) * n_cols_per_param)
        block_boxes = [axes_all[0, c].get_position() for c in cols]

        # horizontal center of the entire block
        x0 = min(b.x0 for b in block_boxes)
        x1 = max(b.x0 + b.width for b in block_boxes)
        x_center = (x0 + x1) / 2
        width = x1 - x0
        # vertical position above top row
        y_top = max(b.y1 for b in block_boxes) + 0.05  # adjust as needed

        # add parameter title
        if param == 'Dp':
            param = 'D*'
        fig.text(x_center, y_top, param, fontsize=20, weight='bold', ha='center', va='bottom')

        y_bottom = min(b.y0 for b in block_boxes) - 0.03  # adjust spacing below
        cbar_height = 0.015  # thickness
        cbar_ax = fig.add_axes([x0+0.01, 0.075, width-0.03, cbar_height])
        if param == 'D*':
            param = 'Dp'
        vmin, vmax = color_ranges[param]
        sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=vmin, vmax=vmax))
        cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
        cbar.set_label(colorbar_labels[param], fontsize=16)
        cbar.ax.tick_params(labelsize=16)

    plt.draw()
    color_subplot_backgrounds(fig, axes_all, algorithm_categories, algorithm_numbers)
    plt.savefig(os.path.join(base_dir, "IVIM_ParameterMaps_3ColumnGrid_FixedScale_NaNsBlack.png"), dpi=300)
    plt.show()

# ==========================================
# 5. Run
# ==========================================
plot_3col_grid(volumes, slice_index=15, orientation='axial')
