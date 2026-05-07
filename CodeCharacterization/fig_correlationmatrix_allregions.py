"""
Script that plots a correlation matrix for all regions separately. The correlation matrix shows the correlations without harmonization.
"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import matplotlib.patches as patches

# ========================================
# ONLY ADDITION: GLOBAL FONT CONTROL
# ========================================
plt.rcParams.update({
    "font.size": 24,
    "axes.titlesize": 24,
    "axes.labelsize": 20,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14
})

# ========================================
# 0. FIXED ALGORITHM ORDER
# ========================================
ALGORITHM_ORDER = [
    'TCML_TechnionIIT_lsqlm',
    'TCML_TechnionIIT_lsqtrf',
    'TCML_TechnionIIT_lsq_sls_lm',
    'TCML_TechnionIIT_lsqBOBYQA',
    'TCML_TechnionIIT_lsq_sls_trf',
    'TCML_TechnionIIT_lsq_sls_BOBYQA',
    'ASD_MemorialSloanKettering_QAMPER_IVIM',
    "IAR_LU_biexp",
    "OGC_AmsterdamUMC_biexp",

    'IAR_LU_modified_mix',
    'IAR_LU_modified_topopro',

    'ETP_SRI_LinearFitting',

    "TF_reference_IVIMfit",
    'PvH_KB_NKI_IVIMfit',

    'TCML_TechnionIIT_SLS',
    "IAR_LU_segmented_2step",
    "IAR_LU_segmented_3step",
    "IAR_LU_subtracted",
    'OGC_AmsterdamUMC_biexp_segmented',
    'PV_MUMC_biexp',
    "OJ_GU_seg",
    'OJ_GU_segMATLAB',

    'OGC_AmsterdamUMC_Bayesian_biexp',
    'OJ_GU_bayesMATLAB',

    'IVIM_NEToptim',
    'Super_IVIM_DC'
]

ALGO_TO_ID = {a: i + 1 for i, a in enumerate(ALGORITHM_ORDER)}
ID_TO_ALGO = {v: k for k, v in ALGO_TO_ID.items()}
ALGO_IDS_ORDERED = [ALGO_TO_ID[a] for a in ALGORITHM_ORDER]


# ========================================
# 1. Harmonization filtering
# ========================================
def filter_algorithms_by_harmonization(df, harmonization_step):

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
        return list(algo_categories.keys())
    elif harmonization_step == 'initialguess_harmonized':
        return [a for a, c in algo_categories.items() if c in [2, 3]]
    elif harmonization_step == 'bounds_and_initialguess_harmonized':
        return [a for a, c in algo_categories.items() if c == 3]
    else:
        return list(algo_categories.keys())


# ========================================
# 2. CATEGORY BANDS
# ========================================
def add_category_bands_heatmap(ax, algorithm_categories, orientation='x'):

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

        rect = patches.Rectangle(
            (start_pos, n_algos) if orientation == 'x' else (-band_thickness, start_pos),
            len(algos) if orientation == 'x' else band_thickness,
            band_thickness if orientation == 'x' else len(algos),
            facecolor=color,
            alpha=0.3,
            transform=ax.transData,
            clip_on=False
        )

        ax.add_patch(rect)
        start_pos += len(algos)


# ========================================
# 3. SINGLE REGION PLOT
# ========================================
def plot_correlation_single(df_region, snr, algorithm_categories):

    params = ['D', 'f', 'Dp']
    algorithms_ordered = ALGO_IDS_ORDERED
    n_algos = len(algorithms_ordered)

    # ONLY CHANGE: larger figure
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    last_hm = None

    for ax, param in zip(axes, params):

        df_filt = df_region[df_region['SNR'] == snr]

        pivot = df_filt.pivot_table(
            index='VoxelID',
            columns='Algorithm',
            values=f'{param}_fitted'
        )

        pivot = pivot.reindex(columns=algorithms_ordered)

        corr = pivot.corr().reindex(
            index=algorithms_ordered,
            columns=algorithms_ordered
        )

        np.fill_diagonal(corr.values, np.nan)

        hm = sns.heatmap(
            corr,
            cmap='viridis',
            vmin=0, vmax=1,
            square=True,
            cbar=False,
            ax=ax,
            mask=corr.isna()  # <- ensures diagonal is white
        )
        last_hm = hm

        ax.set_xticks(np.arange(n_algos) + 0.5)
        ax.set_yticks(np.arange(n_algos) + 0.5)

        ax.set_xticklabels(algorithms_ordered, rotation=90, fontsize=8)
        ax.set_yticklabels(algorithms_ordered, rotation=0, fontsize=8)

        boundaries = np.cumsum([len(v) for v in algorithm_categories.values()])
        for b in boundaries[:-1]:
            ax.axhline(b, color='white', linestyle='--', linewidth=1)
            ax.axvline(b, color='white', linestyle='--', linewidth=1)

        add_category_bands_heatmap(ax, algorithm_categories, 'x')
        add_category_bands_heatmap(ax, algorithm_categories, 'y')

        ax.set_title(param if param != 'Dp' else 'D*')

    cbar_ax = fig.add_axes([0.25, 0.05, 0.5, 0.03])
    cbar = fig.colorbar(last_hm.collections[0], cax=cbar_ax, orientation='horizontal')
    cbar.set_label("Pearson correlation coefficient", fontsize=20)

    plt.tight_layout()
    fig.subplots_adjust(bottom=0.25, top=0.90)

    return fig


# ========================================
# 4. COMBINED PLOT
# ========================================
def plot_correlation_combined(df, snr, algorithm_categories):

    params = ['D', 'f', 'Dp']
    regions = sorted(df["Region"].unique())

    algorithms_ordered = ALGO_IDS_ORDERED
    n_algos = len(algorithms_ordered)

    # ONLY CHANGE: slightly larger figure
    fig, axes = plt.subplots(
        3, len(regions),
        figsize=(5 * len(regions), 12),
        squeeze=False
    )

    last_hm = None

    for i, region in enumerate(regions):
        df_region = df[df["Region"] == region]

        for j, param in enumerate(params):

            ax = axes[j, i]

            pivot = df_region[df_region['SNR'] == snr].pivot_table(
                index='VoxelID',
                columns='Algorithm',
                values=f'{param}_fitted'
            )

            pivot = pivot.reindex(columns=algorithms_ordered)

            corr = pivot.corr().reindex(
                index=algorithms_ordered,
                columns=algorithms_ordered
            )

            hm = sns.heatmap(
                corr,
                cmap='viridis',
                vmin=0, vmax=1,
                square=True,
                cbar=False,
                ax=ax
            )
            last_hm = hm

            ax.set_xticks(np.arange(n_algos) + 0.5)
            ax.set_yticks(np.arange(n_algos) + 0.5)

            ax.set_xticklabels(algorithms_ordered, rotation=90, fontsize=8)
            ax.set_yticklabels(algorithms_ordered, rotation=0, fontsize=8)

            boundaries = np.cumsum([len(v) for v in algorithm_categories.values()])
            for b in boundaries[:-1]:
                ax.axhline(b, color='white', linestyle='--', linewidth=1)
                ax.axvline(b, color='white', linestyle='--', linewidth=1)

            add_category_bands_heatmap(ax, algorithm_categories, 'x')
            add_category_bands_heatmap(ax, algorithm_categories, 'y')

            if j == 0:
                ax.set_title(region)
            if i == 0:
                ax.set_ylabel(param if param != 'Dp' else 'D*', fontsize=14)

    cbar_ax = fig.add_axes([0.25, 0.05, 0.5, 0.02])
    fig.colorbar(last_hm.collections[0], cax=cbar_ax, orientation='horizontal')

    plt.tight_layout(rect=[0, 0.08, 1, 1])
    return fig


# ========================================
# 5. LOAD DATA + RUN (UNCHANGED)
# ========================================
SNR = 20
harmonization_step = "no_harmonization"

file_path = r'C:\TF_IVIM_OSIPI\TF2.4_IVIM-MRI_CodeCollection\test_output_no_harmonization_SNR20_corrected.csv'

df = pd.read_csv(file_path, low_memory=False)
df = df.dropna(subset=["Region"])

for col in ['D_fitted', 'f_fitted', 'Dp_fitted']:
    df[col] = (
        df[col].astype(str)
        .str.replace('[\[\]]', '', regex=True)
        .replace('nan', np.nan)
        .astype(float)
    )

df['Algorithm'] = df['Algorithm'].map(ALGO_TO_ID).astype(int)

if 'VoxelID' not in df.columns:
    df['VoxelID'] = df.groupby(['Region', 'SNR', 'Algorithm']).cumcount()


# ========================================
# CATEGORY MAP (UNCHANGED)
# ========================================
algorithm_categories = {
    'Nonlinear LS': list(range(1, 10)),
    'Variable Projection': [10, 11],
    'Linear LS': [12],
    'Segmented Linear LS': [13, 14],
    'Segmented Nonlinear LS': list(range(15, 23)),
    'Bayesian': [23, 24],
    'Neural network': [25, 26]
}


# ========================================
# FILTER
# ========================================
algorithms_filtered = filter_algorithms_by_harmonization(df, harmonization_step)

algorithm_categories_filtered = {
    cat: [a for a in algos if a in algorithms_filtered]
    for cat, algos in algorithm_categories.items()
}
algorithm_categories_filtered = {k: v for k, v in algorithm_categories_filtered.items() if v}


# ========================================
# RUN
# ========================================
regions = sorted(df["Region"].unique())

for region in regions:
    print(f"Processing region: {region}")

    df_region = df[df["Region"] == region].copy()
    if df_region.empty:
        continue

    fig = plot_correlation_single(df_region, SNR, algorithm_categories_filtered)

    save_folder = rf'C:\TF_IVIM_OSIPI\TF2.4_IVIM-MRI_CodeCollection\CodeCharacterization\{harmonization_step}\{region}/'
    os.makedirs(save_folder, exist_ok=True)

    fig.savefig(os.path.join(save_folder, f'correlation_{region}_SNR{SNR}_whitediagonal.png'),
                dpi=300, bbox_inches='tight')

    plt.close(fig)


fig_combined = plot_correlation_combined(df, SNR, algorithm_categories_filtered)

save_folder = rf'C:\TF_IVIM_OSIPI\TF2.4_IVIM-MRI_CodeCollection\CodeCharacterization\{harmonization_step}\combined/'
os.makedirs(save_folder, exist_ok=True)

fig_combined.savefig(os.path.join(save_folder, f'correlation_ALL_SNR{SNR}.png'),
                     dpi=300, bbox_inches='tight')

plt.close(fig_combined)

print("Done.")