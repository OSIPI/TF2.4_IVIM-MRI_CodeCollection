"""
This script plots the error distributions for the algorithms which allow for harmonization of the bounds and initial
 guess. The csv files are loaded and for a single region the results are plotted.
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
from matplotlib.ticker import ScalarFormatter

# ============================================================
# GLOBAL PLOT STYLE (BIGGER FONTS)
# ============================================================
plt.rcParams.update({
    'font.size': 28,
    'axes.titlesize': 32,
    'axes.labelsize': 28,
    'xtick.labelsize': 28,
    'ytick.labelsize': 28,
    'legend.fontsize': 28,
    'legend.title_fontsize': 28
})

# ============================================================
# CONFIG
# ============================================================
# REGION = "Pancreas_benign"
SNR = 20

csv_files = {

    "No harmonization": rf"C:\TF_IVIM_OSIPI\TF2.4_IVIM-MRI_CodeCollection/CodeCharacterization/Simulated_Data/SNR{SNR}/no_bounds_no_initialguess/test_output_no_harmonization_SNR{SNR}_corrected_implementation_complete.csv",
    "Initial guess": rf"C:\TF_IVIM_OSIPI\TF2.4_IVIM-MRI_CodeCollection/CodeCharacterization/Simulated_Data/SNR{SNR}/no_bounds_initialguess/test_output_initialguess_harmonized_SNR{SNR}_corrected_implementation_complete.csv",
    "Bounds": rf"C:\TF_IVIM_OSIPI\TF2.4_IVIM-MRI_CodeCollection/CodeCharacterization/Simulated_Data/SNR{SNR}/bounds_no_initialguess/test_output_bounds_harmonized_SNR{SNR}_corrected_implementation_complete.csv",
    "Bounds + Initial guess": rf"C:\TF_IVIM_OSIPI\TF2.4_IVIM-MRI_CodeCollection/CodeCharacterization/Simulated_Data/SNR{SNR}/bounds_initialguess/test_output_bounds_and_initialguess_harmonized_SNR{SNR}_corrected_implementation_complete.csv"
}

harmonization_order = [

    "No harmonization",
    "Initial guess",
    "Bounds",
    "Bounds + Initial guess"
]

# ============================================================
# ALGORITHMS
# ============================================================
algorithm_categories = {
    'Nonlinear LS': [
        'TCML_TechnionIIT_lsqlm','TCML_TechnionIIT_lsqtrf','TCML_TechnionIIT_lsq_sls_lm',
        'TCML_TechnionIIT_lsqBOBYQA','TCML_TechnionIIT_lsq_sls_trf','TCML_TechnionIIT_lsq_sls_BOBYQA',
        'ASD_MemorialSloanKettering_QAMPER_IVIM','IAR_LU_biexp','OGC_AmsterdamUMC_biexp'
    ],
    'Segmented Nonlinear LS': [
        'TCML_TechnionIIT_SLS', 'IAR_LU_segmented_2step', 'IAR_LU_segmented_3step',
        'IAR_LU_subtracted', 'OGC_AmsterdamUMC_biexp_segmented', 'PV_MUMC_biexp',
        'OJ_GU_seg', 'OJ_GU_segMATLAB'
    ],
    'Linear LS': ['ETP_SRI_LinearFitting'],
    'Segmented Linear LS': ['TF_reference_IVIMfit', 'PvH_KB_NKI_IVIMfit'],
    'Variable Projection': ['IAR_LU_modified_mix','IAR_LU_modified_topopro'],
    'Bayesian': ['OGC_AmsterdamUMC_Bayesian_biexp','OJ_GU_bayesMATLAB'],
    'Neural network': ['IVIM_NEToptim','Super_IVIM_DC']
}

algorithms_ordered = [a for cat in algorithm_categories.values() for a in cat]

# ============================================================
# EXCLUDE ALGORITHMS 1 AND 3
# ============================================================

excluded_indices = [1, 3]

algorithms_filtered = [
    algo
    for idx, algo in enumerate(algorithms_ordered, start=1)
    if idx not in excluded_indices
]

filtered_labels = [
    idx
    for idx in range(1, len(algorithms_ordered) + 1)
    if idx not in excluded_indices
]

# ============================================================
# HELPERS
# ============================================================
def preprocess_dataframe(df):
    for col in ['D_fitted','f_fitted','Dp_fitted']:
        df[col] = df[col].astype(str).str.replace(r'[\[\]]','',regex=True)\
            .replace('nan',np.nan).astype(float)

    numeric_cols = ['f','Dp','D','f_fitted','Dp_fitted','D_fitted']
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

    df['f_error'] = df['f_fitted'] - df['f']
    df['Dp_error'] = df['Dp_fitted'] - df['Dp']
    df['D_error'] = df['D_fitted'] - df['D']

    return df


def mask_unsupported_harmonization(df, label):
    df = df.copy()
    for algo in df['Algorithm'].unique():
        df_algo = df[df['Algorithm']==algo]

        uses_bounds = df_algo[['D_use_bounds','Dp_use_bounds','f_use_bounds']].any().any()
        uses_init = df_algo[['D_use_initial_guess','Dp_use_initial_guess','f_use_initial_guess']].any().any()

        supported = True
        if label=="Initial guess": supported = uses_init
        elif label=="Bounds": supported = uses_bounds
        elif label=="Bounds + Initial guess": supported = uses_bounds and uses_init

        if not supported:
            df.loc[df['Algorithm']==algo,['f_fitted','Dp_fitted','D_fitted']] = np.nan

    return df


def get_algorithms_with_all_harmonizations(df, y_col):
    valid = []

    for algo in algorithms_filtered:
        ok = True

        for harm in harmonization_order:
            if df[
                (df['Algorithm'] == algo)
                & (df['Harmonization'] == harm)
            ][y_col].dropna().empty:
                ok = False
                break

        if ok:
            valid.append(algo)

    return valid

def set_ylim_from_medians(ax, df, y_col, factor=2):
    """
    Set the y-axis limits based on the largest median value.
    factor controls how much headroom to leave above the highest median.
    """
    grouped = df.groupby(['Algorithm', 'Harmonization'])[y_col]

    q05 = grouped.quantile(0.05).dropna()
    q95 = grouped.quantile(0.95).dropna()

    if q05.empty or q95.empty:
        return

    ymin = factor * q05.median()
    if ymin>0:
        ymin=0

    ymax = factor * q95.median()
    if ymax < 0:
        ymax = 0
    ax.set_ylim(ymin, ymax)

# ============================================================
# 1. ERROR
# ============================================================


# ============================================================
# LOAD DATA
# ============================================================
dfs=[]
for label,path in csv_files.items():
    df_temp=pd.read_csv(path)
    df_temp=mask_unsupported_harmonization(df_temp,label)
    df_temp['Harmonization']=label
    dfs.append(df_temp)

df=pd.concat(dfs,ignore_index=True)
df=preprocess_dataframe(df)
df=df[(df['SNR']==SNR)]


# ============================================================
# ORIGINAL PLOT error
# ============================================================
y_map={'D':'D_error','f':'f_error','Dp':'Dp_error'}

fig,axes=plt.subplots(3,1,figsize=(36,24),sharex=True)

for idx,(ax,param) in enumerate(zip(axes,['D','f','Dp'])):

    sns.boxplot(
        data=df,
        x='Algorithm',
        y=y_map[param],
        hue='Harmonization',
        order=algorithms_filtered,
        hue_order=harmonization_order,
        ax=ax,
        showfliers=False,
        width=0.6,   # ✅ narrower boxes
        showmeans=True,  # ✅ show means as points on the boxplot
        meanprops={
            "marker": "o",
            "markerfacecolor": "black",
            "markeredgecolor": "black",
            "markersize": 16
        }
    )


    # remove side whitespace
    ax.set_xlim(-0.5, len(algorithms_filtered)-0.5)

    for i in range(len(algorithms_filtered) - 1):
        ax.axvline(i+0.5,color='black',alpha=0.25)

    ax.axhline(0,color='gray',linestyle='--')
    ax.set_yscale('linear')

    if param in ['D', 'Dp']:
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((-3, -3))  # always ×10^-3

        ax.yaxis.set_major_formatter(formatter)
        ax.ticklabel_format(axis='y', style='scientific', scilimits=(-3, -3))

        # make exponent text larger
        ax.yaxis.get_offset_text().set_size(24)

    if param == 'D':
        ax.set_ylabel(r'Error in $D$ (mm$^2$/s)')
    elif param == 'Dp':
        ax.set_ylabel(r'Error in $D^*$ (mm$^2$/s)')
    elif param == 'f':
        ax.set_ylabel(r'Error in $f$ (a.u.)')
    ax.set_title(param)

    ax.set_xticks(range(len(algorithms_filtered)))
    ax.set_xticklabels(filtered_labels)

    if ax!=axes[0]: ax.get_legend().remove()

# axes[0].legend(title='Harmonization')

plt.tight_layout()
plt.savefig(rf'grouped_error_boxplot_SNR{SNR}_all_without_1_3_no_inf.png')


# ============================================================
# FILTERED PLOT (KEEP ORIGINAL NUMBERS)
# ============================================================
valid_algorithms=get_algorithms_with_all_harmonizations(df,'D_error')
df_filtered=df[df['Algorithm'].isin(valid_algorithms)]

fig,axes=plt.subplots(3,1,figsize=(30,20),sharex=True)

for idx,(ax,param) in enumerate(zip(axes,['D','f','Dp'])):

    sns.boxplot(
        data=df_filtered,
        x='Algorithm',
        y=y_map[param],
        hue='Harmonization',
        order=valid_algorithms,
        hue_order=harmonization_order,
        ax=ax,
        showfliers=False,
        width=0.6   # ✅ narrower boxes
    )

    ax.set_xlim(-0.5, len(valid_algorithms)-0.5)

    for i in range(len(valid_algorithms)-1):
        ax.axvline(i+0.5,color='black',alpha=0.25)

    ax.axhline(0,color='gray',linestyle='--')
    ax.set_yscale('linear')
    if param == 'D':
        ax.set_ylabel(r'Error in $D$ (mm$^2$/s)')
    elif param == 'Dp':
        ax.set_ylabel(r'Error in $D^*$ (mm$^2$/s)')
    elif param == 'f':
        ax.set_ylabel(r'Error in $f$ (a.u.)')
    title_map = {
        'D': 'D',
        'f': 'f',
        'Dp': 'D*'
    }
    ax.set_title(title_map[param])

    original_indices = [
        algorithms_ordered.index(a) + 1
        for a in valid_algorithms
        if algorithms_ordered.index(a) + 1 not in excluded_indices
    ]

    ax.set_xticks(range(len(valid_algorithms)))
    ax.set_xticklabels(original_indices)

    if ax.get_legend() is not None:
        ax.get_legend().remove()

handles, labels = axes[0].get_legend_handles_labels()

# ============================================================
# FIXED LAYOUT SPACE (KEY CHANGE)
# ============================================================
plt.subplots_adjust(
    hspace=0.35,   # vertical spacing between rows
    left=0.1,     # left margin
    right=0.75,    # IMPORTANT: leave space for legend
    top=0.95,
    bottom=0.10
)

fig.legend(
    handles,
    labels,
    title='Harmonization',
    loc='center right',
    bbox_to_anchor=(0.98, 0.5),
    frameon=True
)

# os.makedirs(rf'C:\TF_IVIM_OSIPI\TF2.4_IVIM-MRI_CodeCollection/CodeCharacterization/GroupedPlots/{REGION}', exist_ok=True)
plt.savefig(rf'grouped_error_boxplot_SNR{SNR}_subset_without_1_3_no_inf.png')


# # ============================================================
# # 2. SQUARED ERROR
# # ============================================================
#
#
# # ============================================================
# # LOAD DATA
# # ============================================================
# dfs=[]
# for label,path in csv_files.items():
#     df_temp=pd.read_csv(path)
#     df_temp=mask_unsupported_harmonization(df_temp,label)
#     df_temp['Harmonization']=label
#     dfs.append(df_temp)
#
# df=pd.concat(dfs,ignore_index=True)
# df=preprocess_dataframe(df)
# df=df[(df['SNR']==SNR)]
#
# # Compute bias per group (using transform to broadcast back to rows)
# df['f_bias'] = df.groupby(['Algorithm', 'SNR', 'Harmonization'])['f_error'].transform('mean')
# df['Dp_bias'] = df.groupby(['Algorithm', 'SNR', 'Harmonization'])['Dp_error'].transform('mean')
# df['D_bias'] = df.groupby(['Algorithm', 'SNR', 'Harmonization'])['D_error'].transform('mean')
#
# # Compute squared deviations for variance boxplots
# df['f_var'] = (df['f_error'] - df['f_bias']) ** 2
# df['Dp_var'] = (df['Dp_error'] - df['Dp_bias']) ** 2
# df['D_var'] = (df['D_error'] - df['D_bias']) ** 2
#
# # Compute absolute errors for RMSE boxplots (individual contributions)
# df['f_squared_error'] = np.square(df['f_error'])
# df['Dp_squared_error'] = np.square(df['Dp_error'])
# df['D_squared_error'] = np.square(df['D_error'])
#
#
# # ============================================================
# # ORIGINAL PLOT error
# # ============================================================
# y_map={'D':'D_squared_error','f':'f_squared_error','Dp':'Dp_squared_error'}
#
# fig,axes=plt.subplots(3,1,figsize=(36,24),sharex=True)
#
# for idx,(ax,param) in enumerate(zip(axes,['D','f','Dp'])):
#
#     sns.boxplot(
#         data=df,
#         x='Algorithm',
#         y=y_map[param],
#         hue='Harmonization',
#         order=algorithms_filtered,
#         hue_order=harmonization_order,
#         ax=ax,
#         showfliers=False,
#         width=0.6,   # ✅ narrower boxes
#         showmeans=True  # ✅ show means as points on the boxplot
#     )
#
#     # remove side whitespace
#     ax.set_xlim(-0.5, len(algorithms_filtered) - 0.5)
#
#     for i in range(len(algorithms_filtered) - 1):
#         ax.axvline(i+0.5,color='black',alpha=0.25)
#
#     ax.axhline(0,color='gray',linestyle='--')
#     # set_ylim_from_medians(ax, df, y_map[param], factor=2)
#     ax.set_yscale('log')
#     if param == 'D':
#         ax.set_ylabel(r'Squared error in $D$ (mm$^4$/s$^2$)')
#     elif param == 'Dp':
#         ax.set_ylabel(r'Squared error  in $D^*$ (mm$^4$/s$^2$)')
#     elif param == 'f':
#         ax.set_ylabel(r'Squared error  in $f$ (a.u.)')
#     ax.set_title(param)
#
#     ax.set_xticks(range(len(algorithms_filtered)))
#     ax.set_xticklabels(filtered_labels)
#
#     if ax!=axes[0]: ax.get_legend().remove()
#
# # axes[0].legend(title='Harmonization')
#
# plt.tight_layout()
# plt.savefig(f'grouped_squarederror_boxplot_SNR{SNR}_all_without_1_3_no_inf_fullbvals_.png')
#
#
# # ============================================================
# # FILTERED PLOT (KEEP ORIGINAL NUMBERS)
# # ============================================================
# valid_algorithms=get_algorithms_with_all_harmonizations(df,'D_squared_error')
# df_filtered=df[df['Algorithm'].isin(valid_algorithms)]
#
# fig,axes=plt.subplots(3,1,figsize=(30,20),sharex=True)
#
# for idx,(ax,param) in enumerate(zip(axes,['D','f','Dp'])):
#
#     sns.boxplot(
#         data=df_filtered,
#         x='Algorithm',
#         y=y_map[param],
#         hue='Harmonization',
#         order=valid_algorithms,
#         hue_order=harmonization_order,
#         ax=ax,
#         showfliers=False,
#         width=0.6   # ✅ narrower boxes
#     )
#
#     ax.set_xlim(-0.5, len(valid_algorithms)-0.5)
#
#     for i in range(len(valid_algorithms)-1):
#         ax.axvline(i+0.5,color='black',alpha=0.25)
#
#     ax.axhline(0,color='gray',linestyle='--')
#     # set_ylim_from_medians(ax, df_filtered, y_map[param], factor=2)
#     ax.set_yscale('log')
#     if param == 'D':
#         ax.set_ylabel(r'Squared error in $D$ (mm$^4$/s$^2$)')
#     elif param == 'Dp':
#         ax.set_ylabel(r'Squared error in $D^*$ (mm$^4$/s$^2$)')
#     elif param == 'f':
#         ax.set_ylabel(r'Squared error in $f$ (a.u.)')
#     title_map = {
#         'D': 'D',
#         'f': 'f',
#         'Dp': 'D*'
#     }
#     ax.set_title(title_map[param])
#
#     original_indices = [
#         algorithms_ordered.index(a) + 1
#         for a in valid_algorithms
#         if algorithms_ordered.index(a) + 1 not in excluded_indices
#     ]
#
#     ax.set_xticks(range(len(valid_algorithms)))
#     ax.set_xticklabels(original_indices)
#
#     if ax.get_legend() is not None:
#         ax.get_legend().remove()
#
# handles, labels = axes[0].get_legend_handles_labels()
#
# # ============================================================
# # FIXED LAYOUT SPACE (KEY CHANGE)
# # ============================================================
# plt.subplots_adjust(
#     hspace=0.35,   # vertical spacing between rows
#     left=0.1,     # left margin
#     right=0.75,    # IMPORTANT: leave space for legend
#     top=0.95,
#     bottom=0.10
# )
#
# fig.legend(
#     handles,
#     labels,
#     title='Harmonization',
#     loc='center right',
#     bbox_to_anchor=(0.98, 0.5),
#     frameon=True
# )
#
# os.makedirs(f'C:\TF_IVIM_OSIPI\TF2.4_IVIM-MRI_CodeCollection/CodeCharacterization/GroupedPlots/{REGION}', exist_ok=True)
# plt.savefig(f'grouped_squarederror_boxplot_SNR{SNR}_subset_without_1_3_no_inf_fullbvals_.png')
#
#
#
# # ============================================================
# # 3. variance
# # ============================================================
#
#
# # ============================================================
# # LOAD DATA
# # ============================================================
# dfs=[]
# for label,path in csv_files.items():
#     df_temp=pd.read_csv(path)
#     df_temp=mask_unsupported_harmonization(df_temp,label)
#     df_temp['Harmonization']=label
#     dfs.append(df_temp)
#
# df=pd.concat(dfs,ignore_index=True)
# df=preprocess_dataframe(df)
# df=df[(df['SNR']==SNR)]
#
# # Compute bias per group (using transform to broadcast back to rows)
# df['f_bias'] = df.groupby(['Algorithm', 'SNR', 'Harmonization'])['f_error'].transform('mean')
# df['Dp_bias'] = df.groupby(['Algorithm', 'SNR', 'Harmonization'])['Dp_error'].transform('mean')
# df['D_bias'] = df.groupby(['Algorithm', 'SNR', 'Harmonization'])['D_error'].transform('mean')
#
# # Compute squared deviations for variance boxplots
# df['f_var'] = (df['f_error'] - df['f_bias']) ** 2
# df['Dp_var'] = (df['Dp_error'] - df['Dp_bias']) ** 2
# df['D_var'] = (df['D_error'] - df['D_bias']) ** 2
#
# # Compute absolute errors for RMSE boxplots (individual contributions)
# df['f_squared_error'] = np.square(df['f_error'])
# df['Dp_squared_error'] = np.square(df['Dp_error'])
# df['D_squared_error'] = np.square(df['D_error'])
#
#
# # ============================================================
# # ORIGINAL PLOT var
# # ============================================================
# y_map={'D':'D_var','f':'f_var','Dp':'Dp_var'}
#
# fig,axes=plt.subplots(3,1,figsize=(36,24),sharex=True)
#
# for idx,(ax,param) in enumerate(zip(axes,['D','f','Dp'])):
#
#     sns.barplot(
#         data=df,
#         x='Algorithm',
#         y=y_map[param],
#         hue='Harmonization',
#         order=algorithms_filtered,
#         hue_order=harmonization_order,
#         ax=ax,
#         errorbar=None,
#         width=0.6)
#
#     # remove side whitespace
#     ax.set_xlim(-0.5, len(algorithms_filtered)-0.5)
#
#     for i in range(len(algorithms_filtered) - 1):
#         ax.axvline(i+0.5,color='black',alpha=0.25)
#
#     ax.axhline(0,color='gray',linestyle='--')
#     # set_ylim_from_medians(ax, df, y_map[param], factor=2)
#     ax.set_yscale('linear')
#
#     if param == 'D':
#         ax.set_ylabel(r'variance in $D$ (mm$^4$/s$^2$)')
#     elif param == 'Dp':
#         ax.set_ylabel(r'variance in $D^*$ (mm$^4$/s$^2$)')
#     elif param == 'f':
#         ax.set_ylabel(r'variance in $f$ (a.u.)')
#     ax.set_title(param)
#
#     ax.set_xticks(range(len(algorithms_filtered)))
#     ax.set_xticklabels(filtered_labels)
#
#     if ax!=axes[0]: ax.get_legend().remove()
#
# # axes[0].legend(title='Harmonization')
#
# plt.tight_layout()
# plt.savefig(rf'grouped_var_barplot_SNR{SNR}_all_without_1_3_no_inf_fullbvals_.png')
#
#
# # ============================================================
# # FILTERED PLOT (KEEP ORIGINAL NUMBERS)
# # ============================================================
# valid_algorithms=get_algorithms_with_all_harmonizations(df,'D_squared_error')
# df_filtered=df[df['Algorithm'].isin(valid_algorithms)]
#
# fig,axes=plt.subplots(3,1,figsize=(30,20),sharex=True)
#
# for idx,(ax,param) in enumerate(zip(axes,['D','f','Dp'])):
#
#     sns.barplot(
#         data=df_filtered,
#         x='Algorithm',
#         y=y_map[param],
#         hue='Harmonization',
#         order=valid_algorithms,
#         hue_order=harmonization_order,
#         ax=ax,
#         errorbar=None,
#         width=0.6
#     )
#
#     ax.set_xlim(-0.5, len(valid_algorithms)-0.5)
#
#     for i in range(len(valid_algorithms)-1):
#         ax.axvline(i+0.5,color='black',alpha=0.25)
#
#     ax.axhline(0,color='gray',linestyle='--')
#     # set_ylim_from_medians(ax, df_filtered, y_map[param], factor=2)
#     ax.set_yscale('linear')
#     if param == 'D':
#         ax.set_ylabel(r'Variance in $D$ (mm$^4$/s$^2$)')
#     elif param == 'Dp':
#         ax.set_ylabel(r'Variance in $D^*$ (mm$^4$/s$^2$)')
#     elif param == 'f':
#         ax.set_ylabel(r'Variance in $f$ (a.u.)')
#     title_map = {
#         'D': 'D',
#         'f': 'f',
#         'Dp': 'D*'
#     }
#     ax.set_title(title_map[param])
#
#     original_indices = [
#         algorithms_ordered.index(a) + 1
#         for a in valid_algorithms
#         if algorithms_ordered.index(a) + 1 not in excluded_indices
#     ]
#
#     ax.set_xticks(range(len(valid_algorithms)))
#     ax.set_xticklabels(original_indices)
#
#     if ax.get_legend() is not None:
#         ax.get_legend().remove()
#
# handles, labels = axes[0].get_legend_handles_labels()
#
# # ============================================================
# # FIXED LAYOUT SPACE (KEY CHANGE)
# # ============================================================
# plt.subplots_adjust(
#     hspace=0.35,   # vertical spacing between rows
#     left=0.1,     # left margin
#     right=0.75,    # IMPORTANT: leave space for legend
#     top=0.95,
#     bottom=0.10
# )
#
# fig.legend(
#     handles,
#     labels,
#     title='Harmonization',
#     loc='center right',
#     bbox_to_anchor=(0.98, 0.5),
#     frameon=True
# )
#
# os.makedirs(rf'C:\TF_IVIM_OSIPI\TF2.4_IVIM-MRI_CodeCollection/CodeCharacterization/GroupedPlots/{REGION}', exist_ok=True)
# plt.savefig(rf'grouped_var_boxplot_SNR{SNR}_subset_without_1_3_no_inf_fullbvals_.png')
#
#
# # ============================================================
# # 3. bias
# # ============================================================
#
#
# # ============================================================
# # LOAD DATA
# # ============================================================
# dfs=[]
# for label,path in csv_files.items():
#     df_temp=pd.read_csv(path)
#     df_temp=mask_unsupported_harmonization(df_temp,label)
#     df_temp['Harmonization']=label
#     dfs.append(df_temp)
#
# df=pd.concat(dfs,ignore_index=True)
# df=preprocess_dataframe(df)
# df=df[(df['SNR']==SNR)]
#
# # Compute bias per group (using transform to broadcast back to rows)
# df['f_bias'] = df.groupby(['Algorithm', 'SNR', 'Harmonization'])['f_error'].transform('mean')
# df['Dp_bias'] = df.groupby(['Algorithm', 'SNR', 'Harmonization'])['Dp_error'].transform('mean')
# df['D_bias'] = df.groupby(['Algorithm', 'SNR', 'Harmonization'])['D_error'].transform('mean')
#
# # Compute squared deviations for bias boxplots
# df['f_var'] = (df['f_error'] - df['f_bias']) ** 2
# df['Dp_var'] = (df['Dp_error'] - df['Dp_bias']) ** 2
# df['D_var'] = (df['D_error'] - df['D_bias']) ** 2
#
# # Compute absolute errors for RMSE boxplots (individual contributions)
# df['f_squared_error'] = np.square(df['f_error'])
# df['Dp_squared_error'] = np.square(df['Dp_error'])
# df['D_squared_error'] = np.square(df['D_error'])
#
#
# # ============================================================
# # ORIGINAL PLOT bias
# # ============================================================
# y_map={'D':'D_bias','f':'f_bias','Dp':'Dp_bias'}
#
# fig,axes=plt.subplots(3,1,figsize=(36,24),sharex=True)
#
# for idx,(ax,param) in enumerate(zip(axes,['D','f','Dp'])):
#
#     sns.barplot(
#         data=df,
#         x='Algorithm',
#         y=y_map[param],
#         hue='Harmonization',
#         order=algorithms_filtered,
#         hue_order=harmonization_order,
#         ax=ax,
#         width=0.6)
#
#     # remove side whitespace
#     ax.set_xlim(-0.5, len(algorithms_filtered)-0.5)
#
#     for i in range(len(algorithms_filtered) - 1):
#         ax.axvline(i+0.5,color='black',alpha=0.25)
#
#     ax.axhline(0,color='gray',linestyle='--')
#     # set_ylim_from_medians(ax, df, y_map[param], factor=2)
#     ax.set_yscale('linear')
#     if param == 'D':
#         ax.set_ylabel(r'bias in $D$ (mm$^2$/s$)')
#     elif param == 'Dp':
#         ax.set_ylabel(r'bias in $D^*$ (mm$^2$/s)')
#     elif param == 'f':
#         ax.set_ylabel(r'bias in $f$ (a.u.)')
#     ax.set_title(param)
#
#     ax.set_xticks(range(len(algorithms_filtered)))
#     ax.set_xticklabels(filtered_labels)
#
#     if ax!=axes[0]: ax.get_legend().remove()
#
# # axes[0].legend(title='Harmonization')
#
# plt.tight_layout()
# plt.savefig(f'grouped_bias_barplot_SNR{SNR}_all_without_1_3_no_inf_fullbvals_.png')
#
#
# # ============================================================
# # FILTERED PLOT (KEEP ORIGINAL NUMBERS)
# # ============================================================
# valid_algorithms=get_algorithms_with_all_harmonizations(df,'D_squared_error')
# df_filtered=df[df['Algorithm'].isin(valid_algorithms)]
#
# fig,axes=plt.subplots(3,1,figsize=(30,20),sharex=True)
#
# for idx,(ax,param) in enumerate(zip(axes,['D','f','Dp'])):
#
#     sns.barplot(
#         data=df_filtered,
#         x='Algorithm',
#         y=y_map[param],
#         hue='Harmonization',
#         order=valid_algorithms,
#         hue_order=harmonization_order,
#         ax=ax,
#         width=0.6
#     )
#
#     ax.set_xlim(-0.5, len(valid_algorithms)-0.5)
#
#     for i in range(len(valid_algorithms)-1):
#         ax.axvline(i+0.5,color='black',alpha=0.25)
#
#     ax.axhline(0,color='gray',linestyle='--')
#     # set_ylim_from_medians(ax, df_filtered, y_map[param], factor=2)
#     ax.set_yscale('linear')
#     if param == 'D':
#         ax.set_ylabel(r'bias in $D$ (mm$^2$/s)')
#     elif param == 'Dp':
#         ax.set_ylabel(r'bias in $D^*$ (mm$^2$/s)')
#     elif param == 'f':
#         ax.set_ylabel(r'bias in $f$ (a.u.)')
#     title_map = {
#         'D': 'D',
#         'f': 'f',
#         'Dp': 'D*'
#     }
#     ax.set_title(title_map[param])
#
#     original_indices = [
#         algorithms_ordered.index(a) + 1
#         for a in valid_algorithms
#         if algorithms_ordered.index(a) + 1 not in excluded_indices
#     ]
#
#     ax.set_xticks(range(len(valid_algorithms)))
#     ax.set_xticklabels(original_indices)
#
#     if ax.get_legend() is not None:
#         ax.get_legend().remove()
#
# handles, labels = axes[0].get_legend_handles_labels()
#
# # ============================================================
# # FIXED LAYOUT SPACE (KEY CHANGE)
# # ============================================================
# plt.subplots_adjust(
#     hspace=0.35,   # vertical spacing between rows
#     left=0.1,     # left margin
#     right=0.75,    # IMPORTANT: leave space for legend
#     top=0.95,
#     bottom=0.10
# )
#
# fig.legend(
#     handles,
#     labels,
#     title='Harmonization',
#     loc='center right',
#     bbox_to_anchor=(0.98, 0.5),
#     frameon=True
# )
#
# os.makedirs(f'C:\TF_IVIM_OSIPI\TF2.4_IVIM-MRI_CodeCollection/CodeCharacterization/GroupedPlots/{REGION}', exist_ok=True)
# plt.savefig(f'grouped_bias_boxplot_SNR{SNR}_subset_without_1_3_no_inf_fullbvals_.png')
#
# # ============================================================
# # SUMMARY TABLE FIGURES
# # ============================================================
#
# import matplotlib.pyplot as plt
#
# metric_maps = {
#     'D': {
#         'Error': 'D_error',
#         'MSE': 'D_squared_error',
#         'Bias': 'D_bias',
#         'Variance': 'D_var'
#     },
#     'f': {
#         'Error': 'f_error',
#         'MSE': 'f_squared_error',
#         'Bias': 'f_bias',
#         'Variance': 'f_var'
#     },
#     'Dstar': {
#         'Error': 'Dp_error',
#         'MSE': 'Dp_squared_error',
#         'Bias': 'Dp_bias',
#         'Variance': 'Dp_var'
#     }
# }
#
# save_folder = rf"C:\TF_IVIM_OSIPI\TF2.4_IVIM-MRI_CodeCollection\CodeCharacterization\GroupedPlots\{REGION}"
# os.makedirs(save_folder, exist_ok=True)
#
# for parameter, metrics in metric_maps.items():
#
#     # --------------------------------------------------------
#     # Compute mean statistics per algorithm/harmonization
#     # --------------------------------------------------------
#     summary = (
#         df.groupby(['Algorithm', 'Harmonization'])
#         .agg({
#             metrics['Error']: 'mean',
#             metrics['MSE']: 'mean',
#             metrics['Bias']: 'mean',
#             metrics['Variance']: 'mean'
#         })
#     )
#
#     summary.columns = ['Error', 'MSE', 'Bias', 'Variance']
#
#     # --------------------------------------------------------
#     # Convert to wide table
#     # --------------------------------------------------------
#     table = summary.unstack('Harmonization')
#
#     # Keep algorithm order
#     table = table.reindex(algorithms_filtered)
#     table.index = [
#         algorithms_ordered.index(a) + 1
#         for a in table.index
#     ]
#     # Desired column order
#     ordered_cols = []
#     for harm in harmonization_order:
#         for metric in ['Error', 'MSE', 'Bias', 'Variance']:
#             ordered_cols.append((metric, harm))
#
#     table = table[ordered_cols]
#
#     # Rename columns
#     table.columns = [
#         f"{harm}\n{metric}"
#         for metric, harm in table.columns
#     ]
#
#     # Round values
#     table = table.round(5)
#
#     # --------------------------------------------------------
#     # Plot table
#     # --------------------------------------------------------
#     fig_height = max(10, len(table) * 0.45)
#     fig_width = 28
#
#     fig, ax = plt.subplots(figsize=(fig_width, fig_height))
#     ax.axis('off')
#
#     tbl = ax.table(
#         cellText=table.values,
#         rowLabels=table.index,
#         colLabels=table.columns,
#         loc='center',
#         cellLoc='center'
#     )
#
#     tbl.auto_set_font_size(False)
#     tbl.set_fontsize(8)
#     tbl.scale(1.2, 1.6)
#
#     # Make header bold
#     for (row, col), cell in tbl.get_celld().items():
#         if row == 0:
#             cell.set_text_props(weight='bold')
#             cell.set_facecolor('#DDDDDD')
#
#     plt.title(
#         f"{parameter} summary statistics",
#         fontsize=20,
#         weight='bold',
#         pad=20
#     )
#
#     plt.tight_layout()
#
#     # plt.savefig(
#     #     os.path.join(save_folder, f"summary_table_{parameter}_SNR{SNR}_without_1_3_no_inf_fullbvals_.png"),
#     #     dpi=300,
#     #     bbox_inches='tight'
#     # )
#
#     plt.close()
#
# print("Summary tables saved.")