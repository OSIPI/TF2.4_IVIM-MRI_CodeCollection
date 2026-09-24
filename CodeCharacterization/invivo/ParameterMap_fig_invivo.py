"""
Script that plots the parameter maps for all implementations
"""
import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import math
import pandas as pd
import seaborn as sns

##Code to plot a single slice of the in-vivo fitting results

# ==========================================
# 1. Setup
# ==========================================
base_dir = r"C:\TF_IVIM_OSIPI\TF2.4_IVIM-MRI_CodeCollection\CodeCharacterization\invivo\results"

parameters = ['D', 'f', 'Dp']
param_file_map = {'D': 'D', 'f': 'f', 'Dp': 'Dp'}

harmonizations = {
    "No harmonization": "no_harmonization",
    "Initial guess": "initialguess_harmonized",
    "Bounds": "bounds_harmonized",
    "Bounds + Initial guess": "bounds_and_initialguess_harmonized"
}

# Algorithm names by category
algorithm_categories = {
    'Nonlinear LS': [
        'TCML_TechnionIIT_lsqlm', 'TCML_TechnionIIT_lsqtrf', 'TCML_TechnionIIT_lsq_sls_lm',
        'TCML_TechnionIIT_lsqBOBYQA', 'TCML_TechnionIIT_lsq_sls_trf', 'TCML_TechnionIIT_lsq_sls_BOBYQA',
        'ASD_MemorialSloanKettering_QAMPER_IVIM', "IAR_LU_biexp",
        "OGC_AmsterdamUMC_biexp"
    ],
    'Segmented Nonlinear LS': [
        'TCML_TechnionIIT_SLS', "IAR_LU_segmented_2step", "IAR_LU_segmented_3step",
        "IAR_LU_subtracted", 'OGC_AmsterdamUMC_biexp_segmented', 'PV_MUMC_biexp',
        "OJ_GU_seg", 'OJ_GU_segMATLAB'
    ],
    'Linear LS': ['ETP_SRI_LinearFitting'],
    'Segmented Linear LS ': ["TF_reference_IVIMfit", 'PvH_KB_NKI_IVIMfit'],
    'Variable Projection': ['IAR_LU_modified_mix', 'IAR_LU_modified_topopro'],
    'Bayesian': ['OGC_AmsterdamUMC_Bayesian_biexp', 'OJ_GU_bayesMATLAB'],
    'Neural network': ['IVIM_NEToptim', 'Super_IVIM_DC']
}

# Flatten all algorithms into a single list for loading
all_algorithms = [algo for cat in algorithm_categories.values() for algo in cat]


# Map each algorithm to a number for plotting
algorithm_numbers = {algo: i+1 for i, algo in enumerate(all_algorithms)}

# ==========================================
# Algorithms to exclude from CV calculations
# ==========================================

excluded_algorithm_ids = [1, 3]

cv_algorithms = [
    algo for algo in all_algorithms
    if algorithm_numbers[algo] not in excluded_algorithm_ids
]

included_algorithms = [
    algo for algo in all_algorithms
    if algorithm_numbers[algo] not in excluded_algorithm_ids
]
n_algos = len(included_algorithms)
ordered_categories = [
    'Nonlinear LS',
    'Segmented Nonlinear LS',
    'Linear LS',
    'Segmented Linear LS ',
    'Variable Projection',
    'Bayesian',
    'Neural network'
]

slice_nr = 12

mask_gray = nib.load(
    r'C:\TF_IVIM_OSIPI\TF2.4_IVIM-MRI_CodeCollection\download\Data\Brain_mask_gray_matter.nii.gz'
).get_fdata().astype(bool)

mask_white = nib.load(
    r'C:\TF_IVIM_OSIPI\TF2.4_IVIM-MRI_CodeCollection\download\Data\Brain_mask_white_matter.nii.gz'
).get_fdata().astype(bool)

mask_gray = mask_gray[:, :, slice_nr]
mask_white = mask_white[:, :, slice_nr]

masks = {
    "Gray Matter": mask_gray,
    "White Matter": mask_white
}
cmap = plt.get_cmap('tab20')
category_colors = {
    cat: cmap(i / len(ordered_categories))
    for i, cat in enumerate(ordered_categories)
}

# ==========================================
# 2. Load volumes
# ==========================================
def load_all_volumes(base_dir, algorithms, parameters, harmonizations):

    data_dict = {}

    for harm_label, harm_prefix in harmonizations.items():

        data_dict[harm_label] = {}

        for algo in algorithms:

            data_dict[harm_label][algo] = {}

            for param in parameters:

                file_param = param_file_map[param]

                nii_path = os.path.join(
                    base_dir,
                    f"{harm_prefix}_{algo}_brain_slice12_fit_{file_param}_new_invivo.nii"
                )

                if os.path.exists(nii_path):

                    data = nib.load(nii_path).get_fdata()

                elif os.path.exists(nii_path + ".gz"):

                    data = nib.load(nii_path + ".gz").get_fdata()

                else:

                    data = None

                    print(f"Missing: {nii_path}")

                data_dict[harm_label][algo][param] = data

    return data_dict

volumes = load_all_volumes(
    base_dir,
    all_algorithms,
    parameters,
    harmonizations
)

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

def color_subplot_backgrounds(
        fig,
        axes,
        algorithm_categories,
        mapping):

    algoid_to_category = {}

    for cat, algos in algorithm_categories.items():

        for algo in algos:

            algo_id = mapping.get(algo)

            if algo_id is not None:
                algoid_to_category[algo_id] = cat

    for i in range(axes.shape[0]):

        for j in range(axes.shape[1]):

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

            color = category_colors.get(
                category,
                (0.8, 0.8, 0.8, 1.0)
            )

            pos = ax.get_position()

            padding_x = 0.01
            padding_y = 0.01

            rect = plt.Rectangle(
                (
                    pos.x0 - padding_x,
                    pos.y0 - padding_y
                ),
                pos.width + 2 * padding_x,
                pos.height + 2 * padding_y,
                transform=fig.transFigure,
                facecolor=color,
                alpha=0.3,
                linewidth=0,
                zorder=-100
            )

            fig.patches.append(rect)


# ==========================================
# 4. Plot — 3-column figure, 6×5 per parameter, NaNs black
# ==========================================
def plot_3col_grid(volumes, harmonization, slice_index=None, orientation='axial'):
    n_rows, n_cols_per_param = 6, 4
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
                algo = included_algorithms[i]
                vol = volumes[harmonization][algo][param]

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
    n_cols_per_param = 4
    block_spacing = 0.03  # fraction of figure width

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
    plt.savefig(os.path.join(base_dir, "IVIM_ParameterMaps_3ColumnGrid_FixedScale_NaNsBlack_no_harmonization_without_1_3_new_invivo.png"), dpi=300)
    plt.show()

# ==========================================
# 5. Run
# ==========================================
plot_3col_grid(volumes, "No harmonization", slice_index=0, orientation='axial')

# ==========================================
# 6. Voxelwise CV maps across algorithms
# ==========================================

def plot_cv_maps(volumes, harm, slice_index=None, orientation='axial'):

    fig, axes = plt.subplots(
        1,
        len(parameters),
        figsize=(15, 5)
    )

    for ax, param in zip(axes, parameters):

        # -----------------------------
        # collect all algorithm volumes
        # -----------------------------
        vol_list = []

        for algo in cv_algorithms:

            vol = volumes[harm][algo][param]

            if vol is not None:
                vol_list.append(vol)

        stack = np.stack(vol_list, axis=0)

        # -----------------------------
        # voxel-wise mean/std
        # -----------------------------
        mean_map = np.nanmean(stack, axis=0)
        std_map = np.nanstd(stack, axis=0)

        cv_map = np.full_like(mean_map, np.nan)

        valid = np.abs(mean_map) > 0

        cv_map[valid] = (
            100 * std_map[valid] /
            np.abs(mean_map[valid])
        )

        # -----------------------------
        # choose slice
        # -----------------------------
        if slice_index is None:
            slice_index = cv_map.shape[2] // 2

        if orientation == 'axial':
            img = np.rot90(cv_map[:, :, slice_index])

        elif orientation == 'coronal':
            img = np.rot90(cv_map[:, slice_index, :])

        elif orientation == 'sagittal':
            img = np.rot90(cv_map[slice_index, :, :])

        else:
            raise ValueError(
                "orientation must be axial, coronal or sagittal"
            )

        # -----------------------------
        # plotting
        # -----------------------------
        masked_img = np.ma.masked_invalid(img)

        cmap = plt.cm.get_cmap("hot").copy()
        cmap.set_bad("black")

        vmax = np.nanpercentile(cv_map, 95)

        im = ax.imshow(
            masked_img,
            cmap=cmap,
            vmin=0,
            vmax=vmax
        )

        title = param
        if param == "Dp":
            title = "D*"

        ax.set_title(
            f"{title}\nVoxel-wise CV (%)",
            fontsize=14
        )

        ax.axis("off")

        cbar = fig.colorbar(
            im,
            ax=ax,
            fraction=0.046,
            pad=0.04
        )

        cbar.set_label("CV (%)")

    plt.tight_layout()

    outfile = os.path.join(
        base_dir,
        "Voxelwise_CV_Maps_NoHarmonization_without_1_3_new_invivo.png"
    )

    plt.savefig(
        outfile,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    print(f"Saved {outfile}")

# ==========================================
# 7. Run CV map plot
# ==========================================

plot_cv_maps(
    volumes,
"No harmonization",
    slice_index=0,
    orientation='axial'
)

# ==========================================
# Category-specific voxelwise CV maps
# ==========================================

def plot_category_cv_maps(
        volumes,
        harm,
        algorithm_categories,
        slice_index=None,
        orientation='axial'):

    valid_categories = [
        cat
        for cat, algos in algorithm_categories.items()
        if len(algos) > 1
    ]

    for param in parameters:

        n_categories = len(valid_categories)

        ncols = 3
        nrows = int(np.ceil(n_categories / ncols))

        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(5 * ncols, 4 * nrows)
        )

        axes = np.array(axes).flatten()

        # =====================================================
        # Compute CV maps first
        # =====================================================

        cv_maps = {}
        all_values = []

        for category in valid_categories:

            algos = [
                algo for algo in algorithm_categories[category]
                if algorithm_numbers[algo] not in excluded_algorithm_ids
            ]

            vol_list = []

            for algo in algos:

                vol = volumes[harm].get(algo, {}).get(param)

                if vol is not None:
                    vol_list.append(vol)

            if len(vol_list) < 2:
                continue

            stack = np.stack(vol_list, axis=0)

            mean_map = np.nanmean(stack, axis=0)
            std_map = np.nanstd(stack, axis=0)

            cv_map = np.full_like(mean_map, np.nan)

            valid = np.abs(mean_map) > 0

            cv_map[valid] = (
                100.0 *
                std_map[valid] /
                np.abs(mean_map[valid])
            )

            cv_maps[category] = cv_map

            all_values.extend(
                cv_map[np.isfinite(cv_map)]
            )

        if len(all_values) == 0:
            print(f"No valid data found for {param}")
            continue

        # Shared color scale
        vmax = np.percentile(all_values, 99)

        # =====================================================
        # Plot each category
        # =====================================================

        im = None

        for ax, category in zip(axes, valid_categories):

            if category not in cv_maps:

                ax.axis("off")
                continue

            cv_map = cv_maps[category]

            if slice_index is None:
                slice_idx = cv_map.shape[2] // 2
            else:
                slice_idx = slice_index

            if orientation == "axial":
                img = np.rot90(cv_map[:, :, slice_idx])

            elif orientation == "coronal":
                img = np.rot90(cv_map[:, slice_idx, :])

            elif orientation == "sagittal":
                img = np.rot90(cv_map[slice_idx, :, :])

            else:
                raise ValueError(
                    "orientation must be axial, coronal, or sagittal"
                )

            masked_img = np.ma.masked_invalid(img)

            cmap = plt.cm.get_cmap("hot").copy()
            cmap.set_bad("black")

            im = ax.imshow(
                masked_img,
                cmap=cmap,
                vmin=0,
                vmax=vmax
            )

            ax.set_title(
                category,
                fontsize=12,
                weight='bold'
            )

            ax.axis("off")

        # Remove unused axes
        for ax in axes[n_categories:]:
            fig.delaxes(ax)

        # =====================================================
        # Layout
        # =====================================================

        display_name = param
        if param == "Dp":
            display_name = "D*"

        fig.suptitle(
            f"{display_name}: Voxel-wise CV Maps by Algorithm Category",
            fontsize=16,
            weight='bold'
        )

        fig.subplots_adjust(
            left=0.03,
            right=0.97,
            top=0.88,
            bottom=0.12,
            wspace=0.05,
            hspace=0.15
        )

        # =====================================================
        # Colorbar
        # =====================================================

        cbar_ax = fig.add_axes([
            0.25,  # left
            0.05,  # bottom
            0.50,  # width
            0.025  # height
        ])

        cbar = fig.colorbar(
            im,
            cax=cbar_ax,
            orientation='horizontal'
        )

        cbar.set_label(
            "Voxel-wise CV (%)",
            fontsize=12
        )

        # =====================================================
        # Save
        # =====================================================

        outfile = os.path.join(
            base_dir,
            f"Category_CV_Maps_{param}_without_1_3_new_invivo.png"
        )

        plt.savefig(
            outfile,
            dpi=300,
            bbox_inches="tight"
        )

        plt.show()

        print(f"Saved {outfile}")

# ==========================================
# Run category CV maps
# ==========================================

plot_category_cv_maps(
    volumes,
    "No harmonization",
    algorithm_categories,
    slice_index=0,
    orientation='axial'
)

def compute_cv_statistics(cv_map, tissue_mask):

    values = cv_map[tissue_mask]

    values = values[np.isfinite(values)]

    if len(values) == 0:
        return None

    return {
        "Mean CV (%)": np.mean(values),
        "Median CV (%)": np.median(values),
        "Std CV (%)": np.std(values),
        "P95 CV (%)": np.percentile(values, 95),
        "N voxels": len(values)
    }

def calculate_tissue_cv_statistics_all_algorithms(volumes, harm):

    results = []

    for param in parameters:

        vol_list = []

        for algo in included_algorithms:

            vol = volumes[harm][algo][param]

            if vol is not None:
                vol_list.append(vol)

        stack = np.stack(vol_list, axis=0)

        mean_map = np.nanmean(stack, axis=0)
        std_map = np.nanstd(stack, axis=0)

        cv_map = np.full_like(mean_map, np.nan)

        valid = np.abs(mean_map) > 0

        cv_map[valid] = (
            100 * std_map[valid]
            / np.abs(mean_map[valid])
        )

        for tissue_name, tissue_mask in masks.items():

            stats = compute_cv_statistics(
                cv_map,
                tissue_mask
            )

            if stats is None:
                continue

            row = {
                "Parameter": param,
                "Tissue": tissue_name
            }

            row.update(stats)

            results.append(row)

    results_df = pd.DataFrame(results)

    outfile = os.path.join(
        base_dir,
        "CV_Statistics_AllImplementations_new_invivo.csv"
    )

    results_df.to_csv(outfile, index=False)

    print(results_df)
    print(f"Saved {outfile}")

def calculate_tissue_cv_statistics_by_category(
        volumes,
        harm,
        algorithm_categories):

    results = []

    for param in parameters:

        for category in algorithm_categories:

            algos = [
                algo
                for algo in algorithm_categories[category]
                if algorithm_numbers[algo]
                not in excluded_algorithm_ids
            ]

            vol_list = []

            for algo in algos:

                vol = volumes[harm].get(algo, {}).get(param)

                if vol is not None:
                    vol_list.append(vol)

            if len(vol_list) < 2:
                continue

            stack = np.stack(vol_list, axis=0)

            mean_map = np.nanmean(stack, axis=0)
            std_map = np.nanstd(stack, axis=0)

            cv_map = np.full_like(mean_map, np.nan)

            valid = np.abs(mean_map) > 0

            cv_map[valid] = (
                100 * std_map[valid]
                / np.abs(mean_map[valid])
            )

            for tissue_name, tissue_mask in masks.items():

                stats = compute_cv_statistics(
                    cv_map,
                    tissue_mask
                )

                if stats is None:
                    continue

                row = {
                    "Category": category,
                    "Parameter": param,
                    "Tissue": tissue_name
                }

                row.update(stats)

                results.append(row)

    results_df = pd.DataFrame(results)

    outfile = os.path.join(
        base_dir,
        "CV_Statistics_ByCategory_new_invivo.csv"
    )

    results_df.to_csv(outfile, index=False)

    print(results_df)
    print(f"Saved {outfile}")

calculate_tissue_cv_statistics_all_algorithms(volumes, "No harmonization")

calculate_tissue_cv_statistics_by_category(volumes, "No harmonization",algorithm_categories)

def summarize_cv(cv_map, tissue_mask):

    if cv_map.ndim == 3:
        cv_map = cv_map[:, :, 0]

    values = cv_map[tissue_mask]

    values = values[np.isfinite(values)]

    if len(values) == 0:
        return None

    return {
        "median": np.median(values),
        "p10": np.percentile(values, 10),
        "p90": np.percentile(values, 90)
    }

def print_tissue_cv_all_implementations(volumes, harm):

    print("\n===== ALL IMPLEMENTATIONS =====")

    for param in parameters:

        vol_list = []

        for algo in included_algorithms:
            vol = volumes[harm][algo][param]

            if vol is not None:
                vol_list.append(vol)

        stack = np.stack(vol_list, axis=0)

        mean_map = np.nanmean(stack, axis=0)
        std_map = np.nanstd(stack, axis=0)

        cv_map = np.full_like(mean_map, np.nan)

        valid = np.abs(mean_map) > 0

        cv_map[valid] = (
            100 * std_map[valid]
            / np.abs(mean_map[valid])
        )

        print(f"\nParameter: {param}")

        for tissue_name, tissue_mask in masks.items():

            stats = summarize_cv(
                cv_map,
                tissue_mask
            )

            print(
                f"{tissue_name}: "
                f"Median={stats['median']:.2f}%, "
                f"P10={stats['p10']:.2f}%, "
                f"P90={stats['p90']:.2f}%"
            )
def print_tissue_cv_by_category(
        volumes,
        harm,
        algorithm_categories):

    print("\n===== CATEGORY ANALYSIS =====")

    for param in parameters:

        print(f"\n\n######## {param} ########")

        for category in algorithm_categories:

            algos = [
                algo
                for algo in algorithm_categories[category]
                if algorithm_numbers[algo]
                not in excluded_algorithm_ids
            ]

            vol_list = []

            for algo in algos:

                vol = volumes[harm].get(algo, {}).get(param)

                if vol is not None:
                    vol_list.append(vol)

            if len(vol_list) < 2:
                continue

            stack = np.stack(vol_list, axis=0)

            mean_map = np.nanmean(stack, axis=0)
            std_map = np.nanstd(stack, axis=0)

            cv_map = np.full_like(mean_map, np.nan)

            valid = np.abs(mean_map) > 0

            cv_map[valid] = (
                100 * std_map[valid]
                / np.abs(mean_map[valid])
            )

            print(f"\n{category}")

            for tissue_name, tissue_mask in masks.items():

                stats = summarize_cv(
                    cv_map,
                    tissue_mask
                )

                print(
                    f"  {tissue_name}: "
                    f"Median={stats['median']:.2f}%, "
                    f"P10={stats['p10']:.2f}%, "
                    f"P90={stats['p90']:.2f}%"
                )
print_tissue_cv_all_implementations(volumes, "No harmonization")

print_tissue_cv_by_category(
    volumes,
"No harmonization",
    algorithm_categories
)

def plot_combined_cv_boxplots(volumes, harm, algorithm_categories):

    rows = []

    # ==========================================
    # All implementations
    # ==========================================

    for param in parameters:

        vol_list = []

        for algo in included_algorithms:

            vol = volumes[harm][algo][param]

            if vol is not None:
                vol_list.append(vol)

        stack = np.stack(vol_list, axis=0)

        mean_map = np.nanmean(stack, axis=0)
        std_map = np.nanstd(stack, axis=0)

        cv_map = np.full_like(mean_map, np.nan)

        valid = np.abs(mean_map) > 0

        cv_map[valid] = (
            100 * std_map[valid]
            / np.abs(mean_map[valid])
        )

        if cv_map.ndim == 3:
            cv_map = cv_map[:, :, 0]

        for tissue_name, tissue_mask in masks.items():

            values = cv_map[tissue_mask]
            values = values[np.isfinite(values)]

            for v in values:

                rows.append({
                    "Parameter": param,
                    "Group": "All",
                    "Tissue": tissue_name,
                    "CV": v
                })

    # ==========================================
    # Categories
    # ==========================================

    for param in parameters:

        for category in algorithm_categories:

            algos = [
                algo
                for algo in algorithm_categories[category]
                if algorithm_numbers[algo]
                not in excluded_algorithm_ids
            ]

            vol_list = []

            for algo in algos:

                vol = volumes[harm].get(algo, {}).get(param)

                if vol is not None:
                    vol_list.append(vol)

            if len(vol_list) < 2:
                continue

            stack = np.stack(vol_list, axis=0)

            mean_map = np.nanmean(stack, axis=0)
            std_map = np.nanstd(stack, axis=0)

            cv_map = np.full_like(mean_map, np.nan)

            valid = np.abs(mean_map) > 0

            cv_map[valid] = (
                100 * std_map[valid]
                / np.abs(mean_map[valid])
            )

            if cv_map.ndim == 3:
                cv_map = cv_map[:, :, 0]

            for tissue_name, tissue_mask in masks.items():

                values = cv_map[tissue_mask]
                values = values[np.isfinite(values)]

                for v in values:

                    rows.append({
                        "Parameter": param,
                        "Group": category,
                        "Tissue": tissue_name,
                        "CV": v
                    })

    df = pd.DataFrame(rows)

    group_order = [
        "All",
        "Nonlinear LS",
        "Segmented Nonlinear LS",
        "Linear LS",
        "Segmented Linear LS",
        "Variable Projection",
        "Bayesian",
        "Neural network"
    ]

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(20, 6),
        sharey=False
    )

    label_map = {
        "D": "D",
        "f": "f",
        "Dp": "Dstar"
    }

    for ax, param in zip(axes, parameters):

        sns.boxplot(
            data=df[df["Parameter"] == param],
            x="Group",
            y="CV",
            hue="Tissue",
            order=group_order,
            showfliers=False,
            whis=(10, 90),
            ax=ax
        )

        ax.set_title(label_map[param], fontsize=16)
        ax.set_ylabel("Voxel-wise CV (%)")
        ax.set_xlabel("")

        ax.tick_params(
            axis="x",
            rotation=45
        )

        if ax != axes[-1]:
            ax.get_legend().remove()

    axes[-1].legend(title="Tissue")

    plt.tight_layout()

    outfile = os.path.join(
        base_dir,
        "Combined_CV_Boxplots_without_1_3_new_invivo.png"
    )

    plt.savefig(
        outfile,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    print(f"Saved {outfile}")

plot_combined_cv_boxplots(
    volumes,
    "No harmonization",
    algorithm_categories
)

def calculate_cv_map(vol_list):

    stack = np.stack(vol_list, axis=0)

    mean_map = np.nanmean(stack, axis=0)
    std_map = np.nanstd(stack, axis=0)

    cv_map = np.full_like(mean_map, np.nan)

    valid = np.abs(mean_map) > 0

    cv_map[valid] = (
        100 * std_map[valid]
        / np.abs(mean_map[valid])
    )

    return cv_map

def build_harmonization_cv_dataframe(volumes):

    rows = []

    for harm in harmonizations:

        # ===========================
        # All implementations
        # ===========================

        for param in parameters:

            vol_list = []

            for algo in included_algorithms:

                vol = volumes[harm][algo][param]

                if vol is not None:
                    vol_list.append(vol)

            cv_map = calculate_cv_map(vol_list)

            if cv_map.ndim == 3:
                cv_map = cv_map[:, :, 0]

            for tissue_name, tissue_mask in masks.items():

                values = cv_map[tissue_mask]
                values = values[np.isfinite(values)]

                for v in values:

                    rows.append({
                        "Harmonization": harm,
                        "Category": "All algorithms",
                        "Parameter": param,
                        "Tissue": tissue_name,
                        "CV": v
                    })

        # ===========================
        # Categories
        # ===========================

        for category in algorithm_categories:

            algos = [
                a for a in algorithm_categories[category]
                if algorithm_numbers[a]
                not in excluded_algorithm_ids
            ]

            if len(algos) < 2:
                continue

            for param in parameters:

                vol_list = []

                for algo in algos:

                    vol = volumes[harm][algo][param]

                    if vol is not None:
                        vol_list.append(vol)

                if len(vol_list) < 2:
                    continue

                cv_map = calculate_cv_map(vol_list)

                if cv_map.ndim == 3:
                    cv_map = cv_map[:, :, 0]

                for tissue_name, tissue_mask in masks.items():

                    values = cv_map[tissue_mask]
                    values = values[np.isfinite(values)]

                    for v in values:

                        rows.append({
                            "Harmonization": harm,
                            "Category": category,
                            "Parameter": param,
                            "Tissue": tissue_name,
                            "CV": v
                        })

    return pd.DataFrame(rows)

df_cv = build_harmonization_cv_dataframe(volumes)

def plot_grouped_harmonization_cv(df_cv):

    category_order = [
        "All algorithms",
        "Nonlinear LS",
        "Segmented Nonlinear LS",
        "Segmented Linear LS ",
        "Variable Projection",
        "Bayesian",
        "Neural network"
    ]

    harmonization_order = [
        "No harmonization",
        "Initial guess",
        "Bounds",
        "Bounds + Initial guess"
    ]

    harm_colors = {
        "No harmonization": "#4C72B0",
        "Initial guess": "#55A868",
        "Bounds": "#C44E52",
        "Bounds + Initial guess": "#DD8452"
    }

    highlight_categories = [
        "All algorithms",
        "Nonlinear LS",
        "Segmented Nonlinear LS"
    ]

    category_labels = [
        f"All algorithms\n(n={len(included_algorithms)})"
    ]

    for cat in category_order[1:]:

        if cat in algorithm_categories:

            n_alg = len([
                a for a in algorithm_categories[cat]
                if algorithm_numbers[a]
                not in excluded_algorithm_ids
            ])

            category_labels.append(
                f"{cat}\n(n={n_alg})"
            )

    parameter_labels = {
        "D": "D",
        "f": "f",
        "Dp": "D*"
    }

    box_width = 0.6
    group_spacing = 0.5

    for param in parameters:

        fig, ax = plt.subplots(figsize=(14, 6))

        n_harm = len(harmonization_order)

        legend_handles = []

        for h_idx, harm in enumerate(harmonization_order):

            data = []
            positions = []

            df_h = df_cv[
                (df_cv["Parameter"] == param) &
                (df_cv["Harmonization"] == harm)
            ]

            for cat_idx, category in enumerate(category_order):

                vals = df_h.loc[
                    df_h["Category"] == category,
                    "CV"
                ].dropna()

                if len(vals) == 0:
                    vals = [np.nan]

                data.append(vals)

                pos = (
                    cat_idx * (n_harm + group_spacing)
                    + h_idx
                )

                positions.append(pos)

            bp = ax.boxplot(
                data,
                positions=positions,
                widths=box_width,
                patch_artist=True,
                showfliers=False,
                whis=(10, 90),
                medianprops={
                    "color": "black",
                    "linewidth": 2.5
                }
            )

            for patch, category in zip(
                    bp["boxes"],
                    category_order):

                patch.set_facecolor(
                    harm_colors[harm]
                )

                if category in highlight_categories:
                    patch.set_alpha(1.0)
                else:
                    patch.set_alpha(0.35)

            legend_handles.append(
                plt.Rectangle(
                    (0, 0),
                    1,
                    1,
                    color=harm_colors[harm],
                    label=harm
                )
            )

        tick_positions = [

            cat_idx * (n_harm + group_spacing)
            + (n_harm - 1) / 2

            for cat_idx in range(
                len(category_order)
            )
        ]

        ax.set_xticks(tick_positions)

        ax.set_xticklabels(
            category_labels,
            rotation=45,
            ha="right"
        )

        ymax = (
            df_cv.loc[
                df_cv["Parameter"] == param,
                "CV"
            ]
            .quantile(0.95)
            * 1.1
        )

        ax.set_ylim(0, ymax)

        ax.set_ylabel(
            "Voxel-wise CV (%)",
            fontsize=14
        )

        ax.set_title(
            f"{parameter_labels[param]}",
            fontsize=18,
            weight="bold"
        )

        ax.legend(
            handles=legend_handles,
            title="Harmonization"
        )

        plt.tight_layout()

        outfile = os.path.join(
            base_dir,
            f"Harmonization_Grouped_CV_{param}_new_invivo.png"
        )

        plt.savefig(
            outfile,
            dpi=300,
            bbox_inches="tight"
        )

        plt.show()

plot_grouped_harmonization_cv(df_cv)


def plot_grouped_harmonization_cv_by_tissue(df_cv):

    category_order = [
        "All algorithms",
        "Nonlinear LS",
        "Segmented Nonlinear LS",
        "Segmented Linear LS ",
        "Variable Projection",
        "Bayesian",
        "Neural network"
    ]

    harmonization_order = [
        "No harmonization",
        "Initial guess",
        "Bounds",
        "Bounds + Initial guess"
    ]

    harm_colors = {
        "No harmonization": "#4C72B0",
        "Initial guess": "#55A868",
        "Bounds": "#C44E52",
        "Bounds + Initial guess": "#DD8452"
    }

    highlight_categories = [
        "All algorithms",
        "Nonlinear LS",
        "Segmented Nonlinear LS"
    ]

    category_labels = [
        f"All algorithms\n(n={len(included_algorithms)})"
    ]

    for cat in category_order[1:]:

        if cat in algorithm_categories:

            n_alg = len([
                a for a in algorithm_categories[cat]
                if algorithm_numbers[a]
                not in excluded_algorithm_ids
            ])

            category_labels.append(
                f"{cat}\n(n={n_alg})"
            )

    parameter_labels = {
        "D": "D",
        "f": "f",
        "Dp": "D*"
    }

    tissues = [
        "Gray Matter",
        "White Matter"
    ]

    box_width = 0.6
    group_spacing = 0.5

    for tissue in tissues:

        for param in parameters:

            fig, ax = plt.subplots(figsize=(14, 6))

            n_harm = len(harmonization_order)

            legend_handles = []

            for h_idx, harm in enumerate(harmonization_order):

                data = []
                positions = []

                df_h = df_cv[
                    (df_cv["Parameter"] == param) &
                    (df_cv["Harmonization"] == harm) &
                    (df_cv["Tissue"] == tissue)
                ]

                for cat_idx, category in enumerate(category_order):

                    vals = df_h.loc[
                        df_h["Category"] == category,
                        "CV"
                    ].dropna()

                    if len(vals) == 0:
                        vals = [np.nan]

                    data.append(vals)

                    pos = (
                        cat_idx * (n_harm + group_spacing)
                        + h_idx
                    )

                    positions.append(pos)

                bp = ax.boxplot(
                    data,
                    positions=positions,
                    widths=box_width,
                    patch_artist=True,
                    showfliers=False,
                    whis=(10, 90),
                    medianprops={
                        "color": "black",
                        "linewidth": 2.5
                    }
                )

                for patch, category in zip(
                        bp["boxes"],
                        category_order):

                    patch.set_facecolor(
                        harm_colors[harm]
                    )

                    if category in highlight_categories:
                        patch.set_alpha(1.0)
                    else:
                        patch.set_alpha(0.35)

                legend_handles.append(
                    plt.Rectangle(
                        (0, 0),
                        1,
                        1,
                        color=harm_colors[harm],
                        label=harm
                    )
                )

            tick_positions = [
                cat_idx * (n_harm + group_spacing)
                + (n_harm - 1) / 2
                for cat_idx in range(len(category_order))
            ]

            ax.set_xticks(tick_positions)

            ax.set_xticklabels(
                category_labels,
                rotation=45,
                ha="right"
            )

            # make highlighted categories bold
            for idx, label in enumerate(ax.get_xticklabels()):

                if category_order[idx] in highlight_categories:
                    label.set_fontweight("bold")

            ymax = (
                df_cv.loc[
                    (df_cv["Parameter"] == param) &
                    (df_cv["Tissue"] == tissue),
                    "CV"
                ]
                .quantile(0.95)
                * 1.1
            )

            ax.set_ylim(0, ymax)

            ax.set_ylabel(
                "Voxel-wise CV (%)",
                fontsize=14
            )

            ax.set_title(
                f"{parameter_labels[param]} - {tissue}",
                fontsize=18,
                weight="bold"
            )

            ax.legend(
                handles=legend_handles,
                title="Harmonization"
            )

            plt.tight_layout()

            outfile = os.path.join(
                base_dir,
                f"Harmonization_Grouped_CV_{param}_{tissue.replace(' ', '_')}_new_invivo.png"
            )

            plt.savefig(
                outfile,
                dpi=300,
                bbox_inches="tight"
            )

            plt.show()

            print(f"Saved {outfile}")

plot_grouped_harmonization_cv_by_tissue(df_cv)