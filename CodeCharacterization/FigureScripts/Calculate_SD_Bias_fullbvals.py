import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)  # all-NaN slices

# ============================================================
# SETTINGS
# ============================================================

SNR = 20
base = r"C:\TF_IVIM_OSIPI\TF2.4_IVIM-MRI_CodeCollection\CodeCharacterization\Simulated_Data"

csv_files = {
    "No harmonization": rf"{base}\SNR{SNR}\no_bounds_no_initialguess\test_output_no_harmonization_SNR{SNR}_corrected_implementation_fullbvals_complete.csv",
    "Initial guess": rf"{base}\SNR{SNR}\no_bounds_initialguess\test_output_initialguess_harmonized_SNR{SNR}_corrected_implementation_fullbvals_complete.csv",
    "Bounds": rf"{base}\SNR{SNR}\bounds_no_initialguess\test_output_bounds_harmonized_SNR{SNR}_corrected_implementation_fullbvals_complete.csv",
    "Bounds + Initial guess": rf"{base}\SNR{SNR}\bounds_initialguess\test_output_bounds_and_initialguess_harmonized_SNR{SNR}_corrected_implementation_fullbvals_complete.csv",
}
harmonization_order = list(csv_files.keys())
EXCLUDED_ALGORITHMS = [
    "TCML_TechnionIIT_lsqlm",
    "TCML_TechnionIIT_lsq_sls_lm",
]
algorithm_categories = {
    'Nonlinear LS': [
        'TCML_TechnionIIT_lsqtrf', 'TCML_TechnionIIT_lsqBOBYQA',
        'TCML_TechnionIIT_lsq_sls_trf', 'TCML_TechnionIIT_lsq_sls_BOBYQA',
        'ASD_MemorialSloanKettering_QAMPER_IVIM', 'IAR_LU_biexp',
        'OGC_AmsterdamUMC_biexp'],
    'Segmented Nonlinear LS': [
        'TCML_TechnionIIT_SLS', 'IAR_LU_segmented_2step', 'IAR_LU_segmented_3step',
        'IAR_LU_subtracted', 'OGC_AmsterdamUMC_biexp_segmented', 'PV_MUMC_biexp',
        'OJ_GU_seg', 'OJ_GU_segMATLAB'],
    'Variable Projection': ['IAR_LU_modified_mix', 'IAR_LU_modified_topopro'],
    'Segmented Linear LS': ['TF_reference_IVIMfit', 'PvH_KB_NKI_IVIMfit'],
    'Bayesian': ['OGC_AmsterdamUMC_Bayesian_biexp', 'OJ_GU_bayesMATLAB'],
    'Neural network': ['IVIM_NEToptim', 'Super_IVIM_DC'],
    'Linear LS': ['ETP_SRI_LinearFitting'],
}

# fitted column -> ground-truth column
parameters = {"D_fitted": "D", "f_fitted": "f", "Dp_fitted": "Dp"}

ORGAN_COL = "Region"   # organ
CURVE_COL = "index"    # signal curve p

# ============================================================
# HELPERS
# ============================================================

def preprocess_dataframe(df):
    for col in ['D_fitted', 'f_fitted', 'Dp_fitted']:
        df[col] = (df[col].astype(str)
                   .str.replace(r'[\[\]]', '', regex=True)
                   .replace('nan', np.nan).astype(float))
    cols = ['f', 'Dp', 'D', 'f_fitted', 'Dp_fitted', 'D_fitted']
    df[cols] = df[cols].apply(pd.to_numeric, errors='coerce')
    return df


def build_arrays(df_h, fitted_col, gt_col):
    """
    Returns
      algos : list of algorithms (rows)
      X     : {organ: array (n_algos, P_organ)}  fitted values x_{n,o,p}
      GT    : {organ: array (P_organ,)}          ground truth x_GT,o,p
    """
    algos = sorted(df_h["Algorithm"].unique())
    X, GT = {}, {}
    for organ, d_o in df_h.groupby(ORGAN_COL):
        piv = d_o.pivot_table(index="Algorithm", columns=CURVE_COL,
                              values=fitted_col, aggfunc="first")
        piv = piv.reindex(algos)
        X[organ] = piv.to_numpy(dtype=float)
        gt = d_o.pivot_table(index="Algorithm", columns=CURVE_COL,
                             values=gt_col, aggfunc="first").reindex(algos)
        GT[organ] = gt.reindex(columns=piv.columns).mean(axis=0).to_numpy(dtype=float)
    return algos, X, GT


# ---------- Eq. 3 / Eq. 2: category-wise SD ----------

def sd_category(X, idx):
    """
    Returns (sd_per_organ {organ: float}, sd_all_organs float)
    idx : row indices of the implementations in the category (or all).
    Eq 3: sqrt( 1/N 1/P sum_n sum_p (x_nop - xbar_op)^2 )   per organ
    Eq 2: sqrt( 1/O sum_o [ 1/N 1/P sum_n sum_p (...)^2 ] ) all organs
    """
    mse_org = {}
    for organ, arr in X.items():
        sub = arr[idx]
        xbar = np.nanmean(sub, axis=0, keepdims=True)      # mean over n
        mse_org[organ] = np.nanmean((sub - xbar) ** 2)     # mean over n,p
    sd_org = {o: np.sqrt(v) for o, v in mse_org.items()}
    sd_all = np.sqrt(np.nanmean(list(mse_org.values())))
    return sd_org, sd_all


# ---------- Eq. 5 / Eq. 4: implementation-specific SD ----------

def sd_implementation(X):
    """
    Deviation of each implementation from the mean of ALL implementations.
    Returns DataFrame (n_algos x organs) for Eq 5 and Series for Eq 4.
    """
    per_org = {}
    for organ, arr in X.items():
        xbar = np.nanmean(arr, axis=0, keepdims=True)      # mean over all n
        per_org[organ] = np.nanmean((arr - xbar) ** 2, axis=1)  # mean over p
    mse = pd.DataFrame(per_org)                            # n x o
    sd_org = np.sqrt(mse)                                  # Eq 5
    sd_all = np.sqrt(mse.mean(axis=1))                     # Eq 4
    return sd_org, sd_all


# ---------- Eq. 6 / 7: error and bias ----------

def bias_matrix(X, GT):
    """b_{n,o} = 1/P sum_p (x_nop - x_GT,op)  -> DataFrame (n_algos x organs)."""
    return pd.DataFrame({o: np.nanmean(X[o] - GT[o][None, :], axis=1) for o in X})


# ---------- Eq. 9 / 8: category-wise RMS bias ----------

def rms_bias_category(B, idx):
    sub = B.iloc[idx]
    rms_org = np.sqrt((sub ** 2).mean(axis=0, skipna=True))      # Eq 9
    rms_all = np.sqrt(np.nanmean(sub.to_numpy() ** 2))           # Eq 8
    return rms_org, rms_all


# ---------- Eq. 10: implementation-wise RMS bias ----------

def rms_bias_implementation(B):
    return np.sqrt((B ** 2).mean(axis=1, skipna=True))


# ============================================================
# LOAD DATA
# ============================================================

dfs = []
for label, path in csv_files.items():
    print(f"Loading {label}")
    t = pd.read_csv(path, low_memory=False)
    t["Harmonization"] = label
    dfs.append(t)
df = preprocess_dataframe(pd.concat(dfs, ignore_index=True))

n_bad = df["Algorithm"].isna().sum()
if n_bad:
    print(f"Dropping {n_bad} rows with missing Algorithm")
df = df.dropna(subset=["Algorithm", "Region", "index"]).copy()
df = df[~df["Algorithm"].isin(EXCLUDED_ALGORITHMS)].copy()

# sanity check: should print 24 and no unmapped algorithms
print("Algorithms remaining:", df["Algorithm"].nunique())
df["Algorithm"] = df["Algorithm"].astype(str)

algo_to_category = {a: c for c, algos in algorithm_categories.items() for a in algos}
df["Algorithm_Category"] = df["Algorithm"].map(algo_to_category)
unmapped = df.loc[df["Algorithm_Category"].isna(), "Algorithm"].unique()
if len(unmapped):
    print("WARNING: algorithms without a category:", unmapped)

# ============================================================
# COMPUTE
# ============================================================

cat_rows, impl_rows = [], []

for harm in harmonization_order:
    print(f"Processing {harm}")
    df_h = df[df["Harmonization"] == harm]

    for fit_col, gt_col in parameters.items():
        algos, X, GT = build_arrays(df_h, fit_col, gt_col)
        algo_cat = [algo_to_category.get(a, np.nan) for a in algos]
        B = bias_matrix(X, GT)
        B.index = algos

        # groups: all algorithms + each category
        groups = {"All algorithms": list(range(len(algos)))}
        for cat in algorithm_categories:
            ids = [i for i, c in enumerate(algo_cat) if c == cat]
            if ids:
                groups[cat] = ids

        # ---- category-wise (Eqs 2, 3, 8, 9) ----
        for cat, ids in groups.items():
            sd_org, sd_all = sd_category(X, ids)
            rms_org, rms_all = rms_bias_category(B, ids)

            cat_rows.append(dict(Harmonization=harm, Parameter=fit_col,
                                 Category=cat, N=len(ids), Organ="All organs",
                                 SD=sd_all, RMS_bias=rms_all))
            for organ in X:
                cat_rows.append(dict(Harmonization=harm, Parameter=fit_col,
                                     Category=cat, N=len(ids), Organ=organ,
                                     SD=sd_org[organ], RMS_bias=rms_org[organ]))

        # ---- implementation-wise (Eqs 4, 5, 10) ----
        sd_imp_org, sd_imp_all = sd_implementation(X)
        rms_imp_all = rms_bias_implementation(B)

        for i, a in enumerate(algos):
            impl_rows.append(dict(Harmonization=harm, Parameter=fit_col,
                                  Algorithm=a, Category=algo_cat[i], Organ="All organs",
                                  SD_vs_all_implementations=sd_imp_all.iloc[i],
                                  RMS_bias=rms_imp_all.iloc[i]))
            for organ in X:
                impl_rows.append(dict(Harmonization=harm, Parameter=fit_col,
                                      Algorithm=a, Category=algo_cat[i], Organ=organ,
                                      SD_vs_all_implementations=sd_imp_org.loc[i, organ],
                                      RMS_bias=abs(B.iloc[i][organ])))  # sqrt(b^2) = |b|

cat_df = pd.DataFrame(cat_rows)
impl_df = pd.DataFrame(impl_rows)

# bias per implementation/organ (Eq 7, signed) for reference
bias_rows = []
for harm in harmonization_order:
    df_h = df[df["Harmonization"] == harm]
    for fit_col, gt_col in parameters.items():
        algos, X, GT = build_arrays(df_h, fit_col, gt_col)
        B = bias_matrix(X, GT)
        B.index = algos
        b = B.stack().reset_index()
        b.columns = ["Algorithm", "Organ", "Bias"]
        b.insert(0, "Parameter", fit_col)
        b.insert(0, "Harmonization", harm)
        bias_rows.append(b)
bias_df = pd.concat(bias_rows, ignore_index=True)

# ============================================================
# EXPORT
# ============================================================

out = f"Variability_Bias_SNR{SNR}_fullbvals.xlsx"
with pd.ExcelWriter(out, engine="openpyxl") as writer:
    cat_df.to_excel(writer, sheet_name="Category-wise (Eq2,3,8,9)", index=False)
    impl_df.to_excel(writer, sheet_name="Implementation (Eq4,5,10)", index=False)
    bias_df.to_excel(writer, sheet_name="Bias per algo+organ (Eq7)", index=False)

print(f"Saved {out}")