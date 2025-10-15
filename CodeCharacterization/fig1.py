import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


def add_category_bands(ax, algorithm_categories, algorithms_ordered, rotation=45):
    """
    Draws background bands per category and labels them below the x-axis.
    Each category gets its own color. Category labels can be rotated.
    """
    # 1. Set algorithm tick labels
    ax.set_xticks(range(len(algorithms_ordered)))
    ax.set_xticklabels(algorithms_ordered, rotation=45, ha='right')

    # 2. Compute category spans
    start_pos = 0
    cat_positions = []
    for cat, algos in algorithm_categories.items():
        end_pos = start_pos + len(algos) - 1
        cat_positions.append((cat, start_pos, end_pos))
        start_pos = end_pos + 1

    # 3. Assign one distinct color per category
    cmap = plt.get_cmap('tab20')
    num_categories = len(cat_positions)

    for i, (cat, x0, x1) in enumerate(cat_positions):
        color = cmap(i / num_categories)  # unique color per category
        ax.axvspan(x0-0.5, x1+0.5, facecolor=color, alpha=0.3, zorder=-1)
        # Label category below x-axis with rotation
        ax.text((x0+x1)/2, -0.08, cat,
                ha='right' if rotation != 0 else 'center', va='top',
                rotation=rotation,
                transform=ax.get_xaxis_transform(),
                fontsize=10, fontweight='bold')

    # 4. Add vertical separators (optional)
    start_pos = 0
    for cat, algos in list(algorithm_categories.items())[:-1]:
        start_pos += len(algos)
        ax.axvline(start_pos - 0.5, color='gray', linestyle='--', linewidth=1)

    # 5. Adjust bottom margin to fit rotated category labels
    ax.figure.subplots_adjust(bottom=0.3)


def create_boxplot(df, y_map, main_title, y_label, log=False):
    # Define algorithm categories and which algorithms belong to each
    algorithm_categories = {
        'Nonlinear Least Squares': ['TCML_TechnionIIT_lsqlm', 'TCML_TechnionIIT_lsqtrf', 'TCML_TechnionIIT_lsq_sls_lm',
                                    'ASD_MemorialSloanKettering_QAMPER_IVIM', "IAR_LU_biexp", 'IAR_LU_modified_mix',
                                    'IAR_LU_modified_topopro', "OGC_AmsterdamUMC_biexp"],
        'Linear fit': ['ETP_SRI_LinearFitting'],
        'Segmented': ['TCML_TechnionIIT_SLS', "IAR_LU_segmented_2step", "IAR_LU_segmented_3step", "IAR_LU_subtracted",
                      'OGC_AmsterdamUMC_biexp_segmented', 'PV_MUMC_biexp', 'PvH_KB_NKI_IVIMfit', "OJ_GU_seg"],
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

    df = df.replace(mapping)

    # Step 2: Update categories to integer IDs
    algorithm_categories = {
        'Nonlinear Least Squares': [1, 2, 3, 4, 5, 6, 7, 8],
        'Linear fit': [9],
        'Segmented': [10, 11, 12, 13, 14, 15, 16, 17],
        'Bayesian': [18, 19],
        'Neural network': [20, 21],
    }

    # Flatten ordered list for x-axis
    algorithms_ordered = []
    for cat, algos in algorithm_categories.items():
        algorithms_ordered.extend(algos)

    # Unique regions
    regions = df['Region'].unique()
    n_regions = len(regions)

    fig, axes = plt.subplots(n_regions, 3, figsize=(18, 6 * n_regions), squeeze=False)
    fig.suptitle(main_title)

    for i, region in enumerate(regions):
        df_region = df[(df['Region'] == region) & (df['SNR'] == 100)]

        for j, param in enumerate(['D', 'f', 'Dp']):
            ax = axes[i, j]

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

            ax.set_title(f'{region} - {param}')
            ax.set_ylabel(y_label)

            # Add category bands below x-axis
            add_category_bands(ax, algorithm_categories, algorithms_ordered)

            # Adjust y-limits **per subplot** based on whiskers
            adjust_ylim_to_box_from_data(
                ax,
                df_region,
                y_col=y_map[param],
                x_col='Algorithm',
                x_order=algorithms_ordered,
                showfliers=False
            )
            # Remove legend if present
            if ax.legend_:
                ax.legend_.remove()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig

def adjust_ylim_to_box_from_data(ax, df, y_col, x_col, x_order, showfliers=False):
    """
    Adjust y-limits based on the actual data used in the boxplot.
    - df: DataFrame filtered for current region/SNR
    - y_col: column used for y-axis
    - x_col: column used for x-axis
    - x_order: list of x-axis values in order
    - showfliers: if False, ignore outliers beyond 1.5*IQR
    """
    y_min = np.inf
    y_max = -np.inf

    for x_val in x_order:
        y_data = df[df[x_col] == x_val][y_col].dropna()
        if len(y_data) == 0:
            continue  # skip if no data for this algorithm

        if not showfliers:
            q1 = np.percentile(y_data, 25)
            q3 = np.percentile(y_data, 75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            y_data = y_data[(y_data >= lower) & (y_data <= upper)]

        if len(y_data) > 0:
            y_min = min(y_min, y_data.min())
            y_max = max(y_max, y_data.max())

    if y_min == np.inf or y_max == -np.inf:
        # fallback if all data is empty
        y_min, y_max = ax.get_ylim()

    padding = (y_max - y_min) * 0.05 if y_max != y_min else 0.1
    ax.set_ylim(y_min - padding, y_max + padding)


# Load the CSV (update the path if needed)
file_path = '/home/rnga/dkuppens/test_output.csv'
df = pd.read_csv(file_path)

# Convert relevant columns to numeric, coercing errors to NaN
numeric_columns = ['f', 'Dp', 'D', 'f_fitted', 'Dp_fitted', 'D_fitted']
for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Create a category column for grouping by SNR/region
# df['category'] = df['Region'] + ' SNR:' + df['SNR'].astype(str)

# Compute errors
df['f_error'] = df['f_fitted'] - df['f']
df['Dp_error'] = df['Dp_fitted'] - df['Dp']
df['D_error'] = df['D_fitted'] - df['D']

# Compute bias per group (using transform to broadcast back to rows)
df['f_bias'] = df.groupby(['Algorithm', 'Region', 'SNR'])['f_error'].transform('mean')
df['Dp_bias'] = df.groupby(['Algorithm', 'Region', 'SNR'])['Dp_error'].transform('mean')
df['D_bias'] = df.groupby(['Algorithm', 'Region', 'SNR'])['D_error'].transform('mean')

# Compute squared deviations for variance boxplots
df['f_var'] = (df['f_error'] - df['f_bias']) ** 2
df['Dp_var'] = (df['Dp_error'] - df['Dp_bias']) ** 2
df['D_var'] = (df['D_error'] - df['D_bias']) ** 2

# Compute absolute errors for RMSE boxplots (individual contributions)
df['f_squared_error'] = np.square(df['f_error'])
df['Dp_squared_error'] = np.square(df['Dp_error'])
df['D_squared_error'] = np.square(df['D_error'])

# 1. Boxplot of E
e_y_map = {'f': 'f_error', 'Dp': 'Dp_error', 'D': 'D_error'}
fig1 = create_boxplot(df, e_y_map, 'Error vs Estimation Method (grouped by category)', 'error', log=False)
plt.savefig('/home/rnga/dkuppens/TF2.4_IVIM-MRI_CodeCollection/CodeCharacterization/nobounds_noinitialguess/e_map.png')
