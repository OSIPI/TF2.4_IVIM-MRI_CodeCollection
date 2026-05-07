"""
Script that plots a correlation matrix. The bottom of the correlation matrix shows the correlations for no harmonization.
 The top of the matrix shows the correlations with full-harmonization.
"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ============================================================
# GLOBAL STYLE (CONSISTENT FONT SYSTEM)
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
# ALGORITHM CATEGORIES
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
# LOAD + PREPROCESS
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
# HYBRID CORRELATION
# ============================================================
def build_hybrid_corr(df_no, df_harm, param):

    pivot_no = df_no.pivot_table(index='VoxelID', columns='Algorithm', values=f'{param}_fitted')
    pivot_harm = df_harm.pivot_table(index='VoxelID', columns='Algorithm', values=f'{param}_fitted')

    pivot_no = pivot_no.reindex(columns=algorithms_ordered)
    pivot_harm = pivot_harm.reindex(columns=algorithms_ordered)

    corr_no = pivot_no.corr()
    corr_harm = pivot_harm.corr()

    hybrid = corr_no.copy()

    for i in range(len(hybrid)):
        for j in range(len(hybrid)):
            if i <= j:
                hybrid.iloc[i, j] = corr_harm.iloc[i, j]

    return hybrid

# ============================================================
# PLOT
# ============================================================
params = ['D', 'f', 'Dp']
fig, axes = plt.subplots(1, 3, figsize=(22, 9))

for ax, param in zip(axes, params):

    hybrid = build_hybrid_corr(df_no, df_harm, param)

    hybrid_plot = hybrid.copy()
    np.fill_diagonal(hybrid_plot.values, np.nan)

    cmap = sns.color_palette("viridis", as_cmap=True)
    cmap.set_bad(color='white')

    hm = sns.heatmap(
        hybrid_plot,
        cmap=cmap,
        vmin=0, vmax=1,
        square=True,
        cbar=False,
        ax=ax,
        linewidths=0.0,
    )

    n = len(hybrid_plot)

    ax.plot([0, n], [0, n], color='white', linewidth=2)

    sizes = [len(v) for v in algorithm_categories.values()]
    for b in np.cumsum(sizes)[:-1]:
        ax.axvline(b, color='gray', linestyle='--', alpha=0.6)
        ax.axhline(b, color='gray', linestyle='--', alpha=0.6)

    add_category_bands(ax, algorithm_categories, n)

    ax.set_xticks(np.arange(n) + 0.5)
    ax.set_yticks(np.arange(n) + 0.5)

    ax.set_xticklabels(algorithms_ordered, rotation=90, fontsize=16)
    ax.set_yticklabels(algorithms_ordered, fontsize=16)

    ax.set_title(param if param != 'Dp' else 'D*', fontsize=26, pad=10)

# ============================================================
# COLORBAR
# ============================================================
cbar_ax = fig.add_axes([0.25, 0.08, 0.5, 0.03])
cbar = fig.colorbar(hm.collections[0], cax=cbar_ax, orientation='horizontal')
cbar.set_label("Correlation", fontsize=22, labelpad=12)
cbar.ax.tick_params(labelsize=18)

plt.tight_layout(rect=[0, 0.12, 1, 1])
plt.show()