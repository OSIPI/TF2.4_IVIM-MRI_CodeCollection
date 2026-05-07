"""
This script plots the error distributions for the algorithms which allow for harmonization of the bounds and initial
 guess. The csv files are loaded and for a single region the results are plotted.
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os

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
REGION = "Pancreas_benign"
SNR = 20

csv_files = {
    "No harmonization": rf"C:\TF_IVIM_OSIPI\TF2.4_IVIM-MRI_CodeCollection\test_output_no_harmonization_SNR{SNR}_corrected.csv",
    "Initial guess": rf"C:\TF_IVIM_OSIPI\TF2.4_IVIM-MRI_CodeCollection\test_output_initialguess_harmonized_SNR{SNR}_corrected.csv",
    "Bounds": rf"C:\TF_IVIM_OSIPI\TF2.4_IVIM-MRI_CodeCollection\test_output_bounds_harmonized_SNR{SNR}_corrected.csv",
    "Bounds + Initial guess": rf"C:\TF_IVIM_OSIPI\TF2.4_IVIM-MRI_CodeCollection\test_output_bounds_and_initialguess_harmonized_SNR{SNR}_corrected.csv"
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
    'Variable Projection': ['IAR_LU_modified_mix','IAR_LU_modified_topopro'],
    'Linear LS': ['ETP_SRI_LinearFitting'],
    'Segmented Linear LS': ['TF_reference_IVIMfit','PvH_KB_NKI_IVIMfit'],
    'Segmented Nonlinear LS': [
        'TCML_TechnionIIT_SLS','IAR_LU_segmented_2step','IAR_LU_segmented_3step',
        'IAR_LU_subtracted','OGC_AmsterdamUMC_biexp_segmented','PV_MUMC_biexp',
        'OJ_GU_seg','OJ_GU_segMATLAB'
    ],
    'Bayesian': ['OGC_AmsterdamUMC_Bayesian_biexp','OJ_GU_bayesMATLAB'],
    'Neural network': ['IVIM_NEToptim','Super_IVIM_DC']
}

algorithms_ordered = [a for cat in algorithm_categories.values() for a in cat]

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
    valid=[]
    for algo in algorithms_ordered:
        ok=True
        for harm in harmonization_order:
            if df[(df['Algorithm']==algo)&(df['Harmonization']==harm)][y_col].dropna().empty:
                ok=False; break
        if ok: valid.append(algo)
    return valid


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
df=df[(df['Region']==REGION)&(df['SNR']==SNR)]

y_map={'D':'D_error','f':'f_error','Dp':'Dp_error'}

# ============================================================
# ORIGINAL PLOT
# ============================================================
fig,axes=plt.subplots(3,1,figsize=(36,24),sharex=True)

for idx,(ax,param) in enumerate(zip(axes,['D','f','Dp'])):

    sns.boxplot(
        data=df,
        x='Algorithm',
        y=y_map[param],
        hue='Harmonization',
        order=algorithms_ordered,
        hue_order=harmonization_order,
        ax=ax,
        showfliers=False,
        width=0.6   # ✅ narrower boxes
    )

    # remove side whitespace
    ax.set_xlim(-0.5, len(algorithms_ordered)-0.5)

    for i in range(len(algorithms_ordered)-1):
        ax.axvline(i+0.5,color='black',alpha=0.25)

    ax.axhline(0,color='gray',linestyle='--')

    if param == 'D':
        ax.set_ylabel(r'Error in $D$ (mm$^2$/s)')
    elif param == 'Dp':
        ax.set_ylabel(r'Error in $D^*$ (mm$^2$/s)')
    elif param == 'f':
        ax.set_ylabel(r'Error in $f$ (a.u.)')
    ax.set_title(param)

    ax.set_xticks(range(len(algorithms_ordered)))
    ax.set_xticklabels([str(i) for i in range(1,len(algorithms_ordered)+1)])

    if ax!=axes[0]: ax.get_legend().remove()

# axes[0].legend(title='Harmonization')

plt.tight_layout()
plt.show()


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

    original_indices=[algorithms_ordered.index(a)+1 for a in valid_algorithms]

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
plt.savefig(rf'C:\TF_IVIM_OSIPI\TF2.4_IVIM-MRI_CodeCollection\CodeCharacterization\GroupedPlots\{REGION}\grouped_error_boxplot_SNR{SNR}_{REGION}_subset.png')
plt.show()