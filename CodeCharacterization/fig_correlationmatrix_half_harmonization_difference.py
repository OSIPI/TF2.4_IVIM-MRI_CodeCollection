"""
Script that plots a correlation matrix. The bottom of the correlation matrix shows the correlations for full harmonization
of all algorithms (bounds and initial guess). The top of the matrix shows the differences between no harmonization and full-harmonization.
"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

# ============================================================
# STYLE
# ============================================================
plt.rcParams.update({
    'font.size': 22,
    'axes.titlesize': 28,
    'axes.labelsize': 24,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'figure.titlesize': 28
})

# ============================================================
# CONFIG
# ============================================================
region = "Pancreas_benign"
SNR = 20

file_no = r"C:\TF_IVIM_OSIPI\TF2.4_IVIM-MRI_CodeCollection/test_output_no_harmonization_SNR20_corrected.csv"
file_harm = r"C:\TF_IVIM_OSIPI\TF2.4_IVIM-MRI_CodeCollection/test_output_bounds_and_initialguess_harmonized_SNR20_corrected.csv"

# ============================================================
# ALGORITHMS
# ============================================================
algorithm_categories = {
    'Nonlinear LS': [
        'TCML_TechnionIIT_lsqlm', 'TCML_TechnionIIT_lsqtrf',
        'TCML_TechnionIIT_lsq_sls_lm', 'TCML_TechnionIIT_lsqBOBYQA',
        'TCML_TechnionIIT_lsq_sls_trf', 'TCML_TechnionIIT_lsq_sls_BOBYQA',
        'ASD_MemorialSloanKettering_QAMPER_IVIM',
        'IAR_LU_biexp',
        'OGC_AmsterdamUMC_biexp'
    ],
    'Variable Projection': ['IAR_LU_modified_mix', 'IAR_LU_modified_topopro'],
    'Linear LS': ['ETP_SRI_LinearFitting'],
    'Segmented Linear LS': ['TF_reference_IVIMfit', 'PvH_KB_NKI_IVIMfit'],
    'Segmented Nonlinear LS': [
        'TCML_TechnionIIT_SLS', 'IAR_LU_segmented_2step',
        'IAR_LU_segmented_3step', 'IAR_LU_subtracted',
        'OGC_AmsterdamUMC_biexp_segmented', 'PV_MUMC_biexp',
        'OJ_GU_seg', 'OJ_GU_segMATLAB'
    ],
    'Bayesian': ['OGC_AmsterdamUMC_Bayesian_biexp', 'OJ_GU_bayesMATLAB'],
    'Neural network': ['IVIM_NEToptim', 'Super_IVIM_DC']
}

algorithms_ordered_names = [a for v in algorithm_categories.values() for a in v]
mapping = {algo: i + 1 for i, algo in enumerate(algorithms_ordered_names)}
algorithms_ordered = list(range(1, len(algorithms_ordered_names) + 1))

# ============================================================
# PREPROCESS
# ============================================================
def preprocess(df):
    for col in ['D_fitted', 'f_fitted', 'Dp_fitted']:
        df[col] = (
            df[col].astype(str)
            .str.replace(r'[\[\]]', '', regex=True)
            .replace('nan', np.nan)
            .astype(float)
        )

    for col in ['f', 'Dp', 'D', 'f_fitted', 'Dp_fitted', 'D_fitted']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    if 'VoxelID' not in df.columns:
        df['VoxelID'] = df.groupby(['Region', 'SNR', 'Algorithm']).cumcount()

    return df


df_no = preprocess(pd.read_csv(file_no))
df_harm = preprocess(pd.read_csv(file_harm))

df_no = df_no[(df_no["Region"] == region) & (df_no["SNR"] == SNR)]
df_harm = df_harm[(df_harm["Region"] == region) & (df_harm["SNR"] == SNR)]

df_no['Algorithm'] = df_no['Algorithm'].map(mapping)
df_harm['Algorithm'] = df_harm['Algorithm'].map(mapping)

df_no.dropna(subset=['Algorithm'], inplace=True)
df_harm.dropna(subset=['Algorithm'], inplace=True)

df_no['Algorithm'] = df_no['Algorithm'].astype(int)
df_harm['Algorithm'] = df_harm['Algorithm'].astype(int)

# ============================================================
# MATRIX
# ============================================================
def build_combined_matrix(df_no, df_harm, param):

    pivot_no = df_no.pivot_table(index='VoxelID', columns='Algorithm', values=f'{param}_fitted')
    pivot_harm = df_harm.pivot_table(index='VoxelID', columns='Algorithm', values=f'{param}_fitted')

    pivot_no = pivot_no.reindex(columns=algorithms_ordered)
    pivot_harm = pivot_harm.reindex(columns=algorithms_ordered)

    corr_no = pivot_no.corr()
    corr_harm = pivot_harm.corr()

    diff = corr_no - corr_harm

    n = len(corr_harm)

    combined = np.zeros_like(corr_harm)

    for i in range(n):
        for j in range(n):
            if i >= j:
                combined[i, j] = corr_harm.iloc[i, j]
            else:
                combined[i, j] = diff.iloc[i, j]

    return combined, corr_harm, diff

# ============================================================
# CATEGORY BANDS
# ============================================================
def add_category_bands(ax, algorithm_categories, n_algos):

    band_thickness = 0.08 * n_algos

    colors = {
        'Nonlinear LS': '#1f77b4',
        'Variable Projection': '#ff7f0e',
        'Linear LS': '#2ca02c',
        'Segmented Linear LS': '#d62728',
        'Segmented Nonlinear LS': '#9467bd',
        'Bayesian': '#8c564b',
        'Neural network': '#e377c2'
    }

    start = 0

    for cat, algos in algorithm_categories.items():
        size = len(algos)
        color = colors.get(cat, '#cccccc')

        # top band
        ax.add_patch(
            patches.Rectangle(
                (start, n_algos),
                size,
                band_thickness,
                facecolor=color,
                alpha=0.3,
                clip_on=False
            )
        )

        # left band
        ax.add_patch(
            patches.Rectangle(
                (-band_thickness, start),
                band_thickness,
                size,
                facecolor=color,
                alpha=0.3,
                clip_on=False
            )
        )

        start += size

# ============================================================
# PLOT
# ============================================================
params = ['D', 'f', 'Dp']
fig, axes = plt.subplots(1, 3, figsize=(22, 11))

for ax, param in zip(axes, params):

    combined, corr_harm, diff = build_combined_matrix(df_no, df_harm, param)
    n = len(combined)

    # =========================
    # DIAGONAL = NaN (WHITE GAP)
    # =========================
    np.fill_diagonal(combined, np.nan)

    mask_top = np.tril(np.ones_like(combined, dtype=bool))
    mask_bottom = np.triu(np.ones_like(combined, dtype=bool), 1)

    cmap_bottom = sns.color_palette("viridis", as_cmap=True)
    cmap_top = sns.color_palette("bwr", as_cmap=True)

    sns.heatmap(
        combined,
        mask=mask_bottom,
        cmap=cmap_bottom,
        vmin=0, vmax=1,
        square=True,
        cbar=False,
        ax=ax
    )

    vmax = np.nanmax(np.abs(diff.values))

    sns.heatmap(
        combined,
        mask=mask_top,
        cmap=cmap_top,
        vmin=-vmax, vmax=vmax,
        square=True,
        cbar=False,
        ax=ax
    )

    # axis limits
    ax.set_xlim(0, n)
    ax.set_ylim(n, 0)

    # ========================================================
    # CATEGORY BOUNDARY LINES (WHITE DASHED)
    # ========================================================
    sizes = [len(v) for v in algorithm_categories.values()]
    boundaries = np.cumsum(sizes)[:-1]

    for b in boundaries:
        ax.axvline(b, color='white', linestyle='--', linewidth=1.5, alpha=0.9, zorder=50)
        ax.axhline(b, color='white', linestyle='--', linewidth=1.5, alpha=0.9, zorder=50)

    # category bands
    add_category_bands(ax, algorithm_categories, n)

    # ticks
    ax.set_xticks(np.arange(n) + 0.5)
    ax.set_yticks(np.arange(n) + 0.5)

    ax.set_xticklabels(algorithms_ordered, rotation=90, ha='center')
    ax.set_yticklabels(algorithms_ordered, rotation=0, ha='right')

    ax.tick_params(axis='both', pad=5)

    ax.set_title(param if param != 'Dp' else 'D*')

# ============================================================
# COLORBARS
# ============================================================
norm_bottom = Normalize(vmin=0, vmax=1)
sm_bottom = ScalarMappable(cmap=sns.color_palette("viridis", as_cmap=True), norm=norm_bottom)
sm_bottom.set_array([])

cbar_ax1 = fig.add_axes([0.25, 0.18, 0.5, 0.02])
fig.colorbar(sm_bottom, cax=cbar_ax1, orientation='horizontal').set_label(
    "Pearson correlation coefficient", fontsize=20
)

all_diffs = []
for param in params:
    _, _, diff = build_combined_matrix(df_no, df_harm, param)
    all_diffs.append(diff.values)

all_diffs = np.concatenate([d.flatten() for d in all_diffs])
vmax = np.nanmax(np.abs(all_diffs))

norm_top = Normalize(vmin=-vmax, vmax=vmax)
sm_top = ScalarMappable(cmap=sns.color_palette("bwr", as_cmap=True), norm=norm_top)
sm_top.set_array([])

cbar_ax2 = fig.add_axes([0.25, 0.07, 0.5, 0.02])
fig.colorbar(sm_top, cax=cbar_ax2, orientation='horizontal').set_label(
    r'$\Delta$ Pearson correlation coefficient (Non-harmonized − Harmonized)',
    fontsize=20
)

plt.tight_layout(rect=[0, 0.14, 1, 1])
plt.show()