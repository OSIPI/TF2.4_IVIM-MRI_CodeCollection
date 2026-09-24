import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

# ============================================================
# SETTINGS
# ============================================================

SNR = 20

csv_files = {
    #"Infinite bounds": rf"C:\TF_IVIM_OSIPI\TF2.4_IVIM-MRI_CodeCollection\CodeCharacterization\Simulated_Data\SNR{SNR}\inf_bounds_no_initialguess\test_output_inf_bounds_SNR{SNR}_corrected_implementation_complete.csv",
    "No harmonization": rf"C:\TF_IVIM_OSIPI\TF2.4_IVIM-MRI_CodeCollection\CodeCharacterization\Simulated_Data\SNR{SNR}\no_bounds_no_initialguess\test_output_no_harmonization_SNR{SNR}_corrected_implementation_complete.csv",
    "Initial guess": rf"C:\TF_IVIM_OSIPI\TF2.4_IVIM-MRI_CodeCollection\CodeCharacterization\Simulated_Data\SNR{SNR}\no_bounds_initialguess\test_output_initialguess_harmonized_SNR{SNR}_corrected_implementation_complete.csv",
    "Bounds": rf"C:\TF_IVIM_OSIPI\TF2.4_IVIM-MRI_CodeCollection\CodeCharacterization\Simulated_Data\SNR{SNR}\bounds_no_initialguess\test_output_bounds_harmonized_SNR{SNR}_corrected_implementation_complete.csv",
    "Bounds + Initial guess": rf"C:\TF_IVIM_OSIPI\TF2.4_IVIM-MRI_CodeCollection\CodeCharacterization\Simulated_Data\SNR{SNR}\bounds_initialguess\test_output_bounds_and_initialguess_harmonized_SNR{SNR}_corrected_implementation_complete.csv"
}

harmonization_order = [
    #"Infinite bounds",
    "No harmonization",
    "Initial guess",
    "Bounds",
    "Bounds + Initial guess"
]

# ============================================================
# ALGORITHM CATEGORIES
# ============================================================

algorithm_categories = {
    'Nonlinear LS': [

        'TCML_TechnionIIT_lsqtrf',

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

    'Variable Projection': [
        'IAR_LU_modified_mix',
        'IAR_LU_modified_topopro'
    ],


    'Segmented Linear LS': [
        'TF_reference_IVIMfit',
        'PvH_KB_NKI_IVIMfit'
    ],

    'Bayesian': [
        'OGC_AmsterdamUMC_Bayesian_biexp',
        'OJ_GU_bayesMATLAB'
    ],

    'Neural network': [
        'IVIM_NEToptim',
        'Super_IVIM_DC'
    ],

    'Linear LS': [
        'ETP_SRI_LinearFitting'
    ]
}

# ============================================================
# HELPERS
# ============================================================

def preprocess_dataframe(df):

    for col in ['D_fitted', 'f_fitted', 'Dp_fitted']:
        df[col] = (
            df[col]
            .astype(str)
            .str.replace(r'[\[\]]', '', regex=True)
            .replace('nan', np.nan)
            .astype(float)
        )

    numeric_cols = [
        'f',
        'Dp',
        'D',
        'f_fitted',
        'Dp_fitted',
        'D_fitted'
    ]

    df[numeric_cols] = df[numeric_cols].apply(
        pd.to_numeric,
        errors='coerce'
    )

    return df


def mask_unsupported_harmonization(df, label):

    df = df.copy()

    for algo in df['Algorithm'].unique():

        df_algo = df[df['Algorithm'] == algo]

        uses_bounds = (
            df_algo[
                ['D_use_bounds',
                 'Dp_use_bounds',
                 'f_use_bounds']
            ]
            .any()
            .any()
        )

        uses_init = (
            df_algo[
                ['D_use_initial_guess',
                 'Dp_use_initial_guess',
                 'f_use_initial_guess']
            ]
            .any()
            .any()
        )

    return df


# ============================================================
# LOAD DATA
# ============================================================

dfs = []

for label, path in csv_files.items():

    print(f"Loading {label}")

    temp = pd.read_csv(path)

    temp = mask_unsupported_harmonization(temp, label)

    temp["Harmonization"] = label

    dfs.append(temp)

df = pd.concat(dfs, ignore_index=True)

df = preprocess_dataframe(df)

# ============================================================
# ADD CATEGORY COLUMN
# ============================================================

algo_to_category = {}

for category, algos in algorithm_categories.items():
    for algo in algos:
        algo_to_category[algo] = category

df["Algorithm_Category"] = df["Algorithm"].map(algo_to_category)

# ============================================================
# CALCULATE CATEGORY VARIABILITY
# ============================================================

parameters = [
    "D_fitted",
    "Dp_fitted",
    "f_fitted"
]

all_results = {}

output_excel = (
    f"AlgorithmCategory_Variability_SNR{SNR}_subset.xlsx"
)

with pd.ExcelWriter(output_excel,
                    engine="openpyxl") as writer:

    for harm in harmonization_order:

        print(f"Processing {harm}")

        df_h = df[
            df["Harmonization"] == harm
        ].copy()

        rows = []

        grouped = df_h.groupby(
            ["Region",
             "index",
             "Algorithm_Category"]
        )

        for (region,
             idx,
             category), group in grouped:

            row = {
                "Region": region,
                "index": idx,
                "Algorithm_Category": category
            }

            for param in parameters:

                values = group[param].dropna()

                n = len(values)

                row[f"{param}_n"] = n

                if n < 2:

                    row[f"{param}_mean"] = np.nan
                    row[f"{param}_sd"] = np.nan
                    row[f"{param}_cv_pct"] = np.nan
                    row[f"{param}_relative_range_pct"] = np.nan

                    continue

                mean_val = values.mean()
                sd_val = values.std(ddof=1)

                denom = abs(mean_val)

                if denom == 0:

                    cv = np.nan
                    rr = np.nan

                else:

                    cv = (
                        100 *
                        sd_val /
                        denom
                    )

                    rr = (
                        100 *
                        (values.max() - values.min())
                        / denom
                    )

                row[f"{param}_mean"] = mean_val
                row[f"{param}_sd"] = sd_val
                row[f"{param}_cv_pct"] = cv
                row[f"{param}_relative_range_pct"] = rr

            rows.append(row)

        results_df = pd.DataFrame(rows)

        results_df.sort_values(
            ["Region",
             "index",
             "Algorithm_Category"],
            inplace=True
        )

        all_results[harm] = results_df

        results_df.to_excel(
            writer,
            sheet_name=harm[:31],
            index=False
        )

print(f"\nSaved {output_excel}")

# ============================================================
# PLOTS: NO HARMONIZATION
# ============================================================

results_df = all_results["No harmonization"]

category_order = (
    ["All algorithms"] +
    [
        cat
        for cat, algos in algorithm_categories.items()
        if len(algos) > 1
    ]
)

parameter_map = {
    "D_fitted": "D_fitted_cv_pct",
    "Dp_fitted": "Dp_fitted_cv_pct",
    "f_fitted": "f_fitted_cv_pct"
}

for parameter_name, cv_col in parameter_map.items():

    plot_df = results_df[
        [
            "Region",
            "Algorithm_Category",
            cv_col
        ]
    ].copy()

    plot_df = plot_df.dropna()

    regions = sorted(
        plot_df["Region"].unique()
    )

    ncols = 3
    nrows = math.ceil(
        len(regions) / ncols
    )

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(18, 4 * nrows),
        sharey=True
    )

    axes = np.array(axes).flatten()

    ymax = (
        plot_df[cv_col]
        .quantile(0.95)
        * 1.1
    )

    for ax, region in zip(axes, regions):

        region_df = plot_df[
            plot_df["Region"] == region
        ]

        data = []
        labels = []

        for category in category_order:

            if category == "All algorithms":

                vals = region_df[cv_col].dropna()

            else:

                vals = region_df.loc[
                    region_df["Algorithm_Category"] == category,
                    cv_col
                ].dropna()

            if len(vals) == 0:
                continue

            data.append(vals)
            labels.append(category)

        if len(data) > 0:

            ax.boxplot(
                data,
                showfliers=False,
                widths=0.6
            )

            ax.set_xticklabels(
                labels,
                rotation=45,
                ha="right"
            )

        ax.set_title(region)

        ax.set_ylabel("CV (%)")

        ax.set_ylim(0, ymax)

    for ax in axes[len(regions):]:
        fig.delaxes(ax)

    fig.suptitle(
        f"No Harmonization\n"
        f"CV Distribution by Algorithm Category\n"
        f"{parameter_name}",
        fontsize=16
    )

    fig.tight_layout()

    filename = (
        f"CV_by_Category_"
        f"{parameter_name}_subset_SNR{SNR}_corrected_implementation.png"
    )

    plt.savefig(
        filename,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Saved {filename}")



# ============================================================
# GLOBAL CV BOXPLOTS
# All regions + all indices combined
# ============================================================

for parameter_name, cv_col in parameter_map.items():

    plot_df = results_df[
        ["Algorithm_Category", cv_col]
    ].copy()

    plot_df = plot_df.dropna()

    data = []
    labels = []

    for category in category_order:

        if category == "All algorithms":

            vals = plot_df[cv_col].dropna()

        else:

            vals = plot_df.loc[
                plot_df["Algorithm_Category"] == category,
                cv_col
            ].dropna()

        if len(vals) == 0:
            continue

        data.append(vals)
        labels.append(category)

    if len(data) == 0:
        continue

    plt.figure(figsize=(10, 6))

    plt.boxplot(
        data,
        labels=labels,
        showfliers=False
    )

    plt.xticks(rotation=45, ha="right")

    plt.ylabel("CV (%)")

    ymax = (
        plot_df[cv_col]
        .quantile(0.95)
        * 1.1
    )

    plt.ylim(0, ymax)

    plt.title(
        f"Distribution of CV Across All Regions and Indices\n"
        f"{parameter_name}"
    )

    plt.tight_layout()

    filename = (
        f"Global_CV_Distribution_{parameter_name}_subset_SNR{SNR}_corrected_implementation.png"
    )

    plt.savefig(
        filename,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Saved {filename}")

print("\nDone.")

# ============================================================
# GROUPED GLOBAL CV BOXPLOTS
# Categories on x-axis, harmonization steps grouped
# ============================================================

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

category_labels = ["All algorithms\n(n={})".format(
    sum(len(v) for v in algorithm_categories.values())
)]

for cat in category_order[1:]:
    category_labels.append(
        f"{cat}\n(n={len(algorithm_categories[cat])})"
    )
box_width = 0.6
group_spacing = 0.5

for parameter_name, cv_col in parameter_map.items():

    fig, ax = plt.subplots(figsize=(12, 6))

    n_harm = len(harmonization_order)

    legend_handles = []

    for h_idx, harm in enumerate(harmonization_order):

        results_df = all_results[harm]

        data = []
        positions = []

        for cat_idx, category in enumerate(category_order):

            if category == "All algorithms":

                vals = results_df[cv_col].dropna()

            else:

                vals = results_df.loc[
                    results_df["Algorithm_Category"] == category,
                    cv_col
                ].dropna()

            if len(vals) == 0:
                vals = [np.nan]

            data.append(vals)

            pos = cat_idx * (n_harm + group_spacing) + h_idx
            positions.append(pos)

        bp = ax.boxplot(
            data,
            positions=positions,
            widths=box_width,
            patch_artist=True,
            showfliers=False,
            medianprops={
                'color': 'black',
                'linewidth': 2.5
                }
        )

        for patch, category in zip(bp["boxes"], category_order):

            patch.set_facecolor(harm_colors[harm])

            if category in highlight_categories:
                patch.set_alpha(1.0)
            else:
                patch.set_alpha(0.35)

        if h_idx == 0:
            legend_handles.append(
                plt.Rectangle((0, 0), 1, 1,
                              color=harm_colors[harm],
                              label=harm)
            )
        else:
            legend_handles.append(
                plt.Rectangle((0, 0), 1, 1,
                              color=harm_colors[harm],
                              label=harm)
            )

    tick_positions = [
        cat_idx * (n_harm + group_spacing) + (n_harm - 1) / 2
        for cat_idx in range(len(category_order))
    ]

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(
        category_labels,
        rotation=45,
        ha="right"
    )

    ymax = max(
        all_results[h][cv_col].quantile(0.95)
        for h in harmonization_order
    ) * 1.1

    ax.set_ylim(0, ymax)
    ax.set_ylabel("CV (%)")

    ax.set_title(
        f"CV Distribution by Algorithm Category\n{parameter_name}"
    )

    ax.legend(handles=legend_handles)

    plt.tight_layout()

    filename = (
        f"Grouped_Global_CV_Distribution_"
        f"{parameter_name}_subset_SNR{SNR}_corrected_implementation.png"
    )

    plt.savefig(
        filename,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Saved {filename}")