import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os

# ========================================
# 1. Helper: Category bands
# ========================================

def add_category_bands(ax, algorithm_categories, algorithms_ordered, category_labels=False, rotation=45):
    ax.set_xticks(range(len(algorithms_ordered)))
    ax.set_xticklabels(algorithms_ordered, fontsize=14)
    ax.tick_params(axis='y', labelsize=14)

    start_pos = 0
    cat_positions = []
    for cat, algos in algorithm_categories.items():
        end_pos = start_pos + len(algos) - 1
        cat_positions.append((cat, start_pos, end_pos))
        start_pos = end_pos + 1

    cmap = plt.get_cmap('tab20')
    num_categories = len(cat_positions)

    for i, (cat, x0, x1) in enumerate(cat_positions):
        color = cmap(i / num_categories)
        ax.axvspan(x0 - 0.5, x1 + 0.5, facecolor=color, alpha=0.3, zorder=-1)

        if category_labels:
            ax.text(
                (x0 + x1) / 2, -0.2, cat,
                ha='right' if rotation != 0 else 'center',
                va='top',
                rotation=rotation,
                transform=ax.get_xaxis_transform(),
                fontsize=14,
                fontweight='bold'
            )

    # separators
    start_pos = 0
    for cat, algos in list(algorithm_categories.items())[:-1]:
        start_pos += len(algos)
        ax.axvline(start_pos - 0.5, color='gray', linestyle='--', linewidth=1)

    ax.figure.subplots_adjust(bottom=0.3)


# ========================================
# 2. Boxplot function
# ========================================

def create_boxplot(df, y_map, main_title, y_label, log=False):

    df = df.dropna(how='all')

    algorithm_categories_names = {
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

    # Map to IDs
    mapping = {}
    counter = 1
    for cat, algos in algorithm_categories_names.items():
        for algo in algos:
            mapping[algo] = counter
            counter += 1

    df = df.replace(mapping)

    algorithm_categories = {
        'Nonlinear LS': list(range(1, 10)),
        'Variable Projection': [10, 11],
        'Linear LS': [12],
        'Segmented Linear LS ': [13, 14],
        'Segmented Nonlinear LS': list(range(15, 23)),
        'Bayesian': [23, 24],
        'Neural network': [25, 26]
    }

    algorithms_ordered = [a for group in algorithm_categories.values() for a in group]

    regions = df['Region'].unique()
    n_regions = len(regions)

    fig, axes = plt.subplots(3, n_regions, figsize=(12*n_regions, 10), squeeze=False)

    for i, region in enumerate(regions):
        df_region = df[df['Region'] == region]

        for j, param in enumerate(['D', 'f', 'Dp']):
            ax = axes[j, i]

            sns.boxplot(
                data=df_region,
                x='Algorithm',
                y=y_map[param],
                ax=ax,
                order=algorithms_ordered,
                showfliers=False
            )

            if log:
                ax.set_yscale('log')

            if param == 'D':
                ax.set_ylabel('error in $D$ (mm²/s)', fontsize=12)
                ax.set_title('$D$')
            elif param == 'Dp':
                ax.set_ylabel('error in $D^*$ (mm²/s)', fontsize=12)
                ax.set_title('$D^*$')
            else:
                ax.set_ylabel('error in $f$ (a.u.)', fontsize=12)
                ax.set_title('$f$')

            ax.set_xlabel('')
            ax.axhline(0, color='gray', linestyle='--', linewidth=1)

            if param in ['D', 'f']:
                add_category_bands(ax, algorithm_categories, algorithms_ordered, category_labels=False)
            else:
                add_category_bands(ax, algorithm_categories, algorithms_ordered, category_labels=True)

            adjust_ylim_to_box_from_data(
                ax,
                df_region,
                y_col=y_map[param],
                x_col='Algorithm',
                x_order=algorithms_ordered,
                showfliers=False
            )

            if ax.legend_:
                ax.legend_.remove()

    plt.tight_layout()
    return fig


# ========================================
# 3. Y-limit helper
# ========================================

def adjust_ylim_to_box_from_data(ax, df, y_col, x_col, x_order, showfliers=False):

    y_min, y_max = np.inf, -np.inf

    for x_val in x_order:
        y_data = df[df[x_col] == x_val][y_col].dropna()

        if len(y_data) == 0:
            continue

        if not showfliers:
            q1 = np.percentile(y_data, 25)
            q3 = np.percentile(y_data, 75)
            iqr = q3 - q1
            y_data = y_data[(y_data >= q1 - 1.5 * iqr) & (y_data <= q3 + 1.5 * iqr)]

        if len(y_data) > 0:
            y_min = min(y_min, y_data.min())
            y_max = max(y_max, y_data.max())

    if y_min == np.inf:
        return

    padding = (y_max - y_min) * 0.05 if y_max != y_min else 0.1
    ax.set_ylim(y_min - padding, y_max + padding)


# ========================================
# 4. Load + preprocess
# ========================================

SNR = 20
file_path = rf'C:\TF_IVIM_OSIPI\TF2.4_IVIM-MRI_CodeCollection\test_output_SNR{SNR}_complete_bounds_initialguess.csv'

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

# Errors
df['f_error'] = df['f_fitted'] - df['f']
df['Dp_error'] = df['Dp_fitted'] - df['Dp']
df['D_error'] = df['D_fitted'] - df['D']

# Bias
df['f_bias'] = df.groupby(['Algorithm', 'Region', 'SNR'])['f_error'].transform('mean')
df['Dp_bias'] = df.groupby(['Algorithm', 'Region', 'SNR'])['Dp_error'].transform('mean')
df['D_bias'] = df.groupby(['Algorithm', 'Region', 'SNR'])['D_error'].transform('mean')

# ========================================
# 5. LOOP OVER ALL REGIONS
# ========================================

regions = df['Region'].dropna().unique()

base_folder = r'C:\TF_IVIM_OSIPI\TF2.4_IVIM-MRI_CodeCollection\CodeCharacterization\bounds_initialguess'

y_map = {'f': 'f_error', 'Dp': 'Dp_error', 'D': 'D_error'}

for region in regions:
    print(f"Processing region: {region}")

    df_region = df[df['Region'] == region]

    if df_region.empty:
        continue

    fig = create_boxplot(
        df_region,
        y_map,
        'Error vs Estimation Method',
        'error',
        log=False
    )

    save_folder = os.path.join(base_folder, region)
    os.makedirs(save_folder, exist_ok=True)

    save_path = os.path.join(save_folder, f'e_map_{region}_SNR{SNR}.png')
    fig.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.close(fig)

print("Done!")