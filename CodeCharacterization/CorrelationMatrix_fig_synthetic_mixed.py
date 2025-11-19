import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as patches

def add_category_bands_heatmap(ax, algorithm_categories, algorithms_ordered, orientation='x'):
    """
    Draw colored bands perfectly aligned with the heatmap cells.
    """
    cmap = plt.get_cmap('tab20')
    num_categories = len(algorithm_categories)
    n_algos = len(algorithms_ordered)

    start_pos = 0
    for i, (cat, algos) in enumerate(algorithm_categories.items()):
        end_pos = start_pos + len(algos) - 1
        color = cmap(i / num_categories)

        if orientation == 'x':
            # rectangle aligned with x-axis cells, below the last row
            rect = patches.Rectangle(
                (start_pos, -0.5 - 0.05 * n_algos + n_algos),  # x, y
                len(algos),  # width
                0.05 * n_algos,  # height
                facecolor=color, alpha=0.3, transform=ax.transData, clip_on=False
            )
        if orientation == 'y':
            # rectangle aligned with y-axis cells, width just left of the matrix
            rect = patches.Rectangle(
                (-0.05 * n_algos, start_pos),  # x, y
                0.05 * n_algos,  # width
                len(algos),  # height
                facecolor=color, alpha=0.3, transform=ax.transData, clip_on=False
            )
        ax.add_patch(rect)
        start_pos = end_pos + 1

    # Separator lines
    start_pos = 0
    for cat, algos in list(algorithm_categories.items())[:-1]:
        start_pos += len(algos)
        if orientation == 'x':
            ax.axvline(start_pos, color='gray', linestyle='--', linewidth=1, zorder=2)
        else:
            ax.axhline(start_pos, color='gray', linestyle='--', linewidth=1, zorder=2)

def plot_correlation_matrix_with_categories(df, param, region, snr, algorithm_categories, mapping, output_path=None):
    df_filtered = df[(df['Region'] == region) & (df['SNR'] == snr)]

    # Algorithm IDs ordered
    algorithms_ordered = []
    for cat, algos in algorithm_categories.items():
        algorithms_ordered.extend([mapping[a] for a in algos if a in mapping])

    pivot_df = df_filtered.pivot_table(
        index='VoxelID',
        columns='Algorithm',
        values=f'{param}_fitted'
    ).dropna(axis=0, how='any')

    if pivot_df.shape[1] < 2:
        print(f"Not enough data for {param}, Region: {region}, SNR: {snr}")
        return None

    corr_matrix = pivot_df.corr().reindex(index=algorithms_ordered, columns=algorithms_ordered)

    # Masks for triangles
    mask_upper = np.tril(np.ones(corr_matrix.shape), k=0).astype(bool)  # lower triangle visible
    mask_lower = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)  # upper triangle visible

    # Labels for upper triangle
    labels = corr_matrix.where(mask_lower).round(2).fillna('')

    fig, ax = plt.subplots(figsize=(12, 10))

    # Lower triangle with colors
    sns.heatmap(
        corr_matrix,
        mask=mask_lower,
        cmap='coolwarm',
        square=True,
        linewidths=1.5,
        cbar_kws={'label': 'Correlation'},
        ax=ax
    )

    # Upper triangle with numbers only
    sns.heatmap(
        corr_matrix,
        mask=mask_upper,
        cmap=ListedColormap(['white']),
        annot=labels,
        fmt='',
        square=True,
        linewidths=1.5,
        cbar=False,
        ax=ax,
        annot_kws={"size": 8}
    )

    # Add category bands
    add_category_bands_heatmap(ax, algorithm_categories, algorithms_ordered, orientation='x')
    add_category_bands_heatmap(ax, algorithm_categories, algorithms_ordered, orientation='y')

    # Extend limits so bands are visible
    ax.set_xlim(-0.5, len(algorithms_ordered) - 0.5)
    ax.set_ylim(len(algorithms_ordered) - 0.5, -0.5)

    ax.set_xlabel("Algorithm ID")
    ax.set_ylabel("Algorithm ID")

    # Optional: remove ticks
    alg_labels = [str(a) for a in algorithms_ordered]  # or use original names

    ax.set_xticks(np.arange(len(algorithms_ordered)) + 0.5)
    ax.set_yticks(np.arange(len(algorithms_ordered)) + 0.5)
    ax.set_xticklabels(alg_labels, rotation=90)  # rotate for readability
    ax.set_yticklabels(alg_labels, rotation=0)

    plt.title(f'Correlation Matrix for {param}\nRegion: {region}, SNR: {snr}')

    if output_path:
        plt.savefig(output_path, dpi=300)
    plt.show()

    return corr_matrix

import pandas as pd

algorithm_categories = {
    'Nonlinear Least Squares': [
        'TCML_TechnionIIT_lsqlm', 'TCML_TechnionIIT_lsqtrf', 'TCML_TechnionIIT_lsq_sls_lm', 'TCML_TechnionIIT_lsqBOBYQA', 'TCML_TechnionIIT_lsq_sls_trf', 'TCML_TechnionIIT_lsq_sls_BOBYQA',
        'ASD_MemorialSloanKettering_QAMPER_IVIM', "IAR_LU_biexp", 'IAR_LU_modified_mix',
        'IAR_LU_modified_topopro', "OGC_AmsterdamUMC_biexp"
    ],
    'Linear fit': ['ETP_SRI_LinearFitting'],
    'Segmented': [
        'TCML_TechnionIIT_SLS', "IAR_LU_segmented_2step", "IAR_LU_segmented_3step",
        "IAR_LU_subtracted", 'OGC_AmsterdamUMC_biexp_segmented', 'PV_MUMC_biexp',
        'PvH_KB_NKI_IVIMfit', "OJ_GU_seg", 'OJ_GU_segMATLAB', 'OJ_GU_bayesMATLAB'
    ],
    'Bayesian': ['OGC_AmsterdamUMC_Bayesian_biexp', "TF_reference_IVIMfit"],
    'Neural network': ['IVIM_NEToptim', 'SUPER_IVIM_DC'],
}

# Step 1: Map algorithm names to integer IDs
mapping = {}
counter = 1
for cat, algos in algorithm_categories.items():
    for algo in algos:
        mapping[algo] = counter
        counter += 1

file_path = r'C:\TF_IVIM_OSIPI\TF2.4_IVIM-MRI_CodeCollection\test_output_SNR20.csv'
df = pd.read_csv(file_path)

# Convert relevant columns to numeric
numeric_columns = ['f', 'Dp', 'D', 'f_fitted', 'Dp_fitted', 'D_fitted']
for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Replace algorithm names with integer IDs
mapping = {}
counter = 1
for cat, algos in algorithm_categories.items():
    for algo in algos:
        mapping[algo] = counter
        counter += 1

df['Algorithm'] = df['Algorithm'].map(mapping)

# Drop rows with unmapped algorithms
df = df.dropna(subset=['Algorithm'])
df['Algorithm'] = df['Algorithm'].astype(int)

# Add VoxelID if missing
if 'VoxelID' not in df.columns:
    df['VoxelID'] = df.groupby(['Region', 'SNR', 'Algorithm']).cumcount()

for region in df['Region'].unique():
    for snr in sorted(df['SNR'].unique()):
        for param in ['D', 'f', 'Dp']:
            output_path = rf'C:\TF_IVIM_OSIPI\TF2.4_IVIM-MRI_CodeCollection\correlationmatrix_{param}_{region}_SNR{snr}_mixed.png'
            plot_correlation_matrix_with_categories(df, param, region, snr, algorithm_categories, mapping, output_path)
