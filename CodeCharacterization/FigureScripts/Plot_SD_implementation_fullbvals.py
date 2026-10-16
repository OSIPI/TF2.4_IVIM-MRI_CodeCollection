import textwrap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D

# ============================================================
# SETTINGS
# ============================================================

SNR = 20

# Output of the SD/bias script (sheet "Implementation (Eq4,5,10)")
INPUT_XLSX = f"Variability_Bias_SNR{SNR}_fullbvals.xlsx"
SHEET = "Implementation (Eq4,5,10)"

# Eq. 4 ("All organs" rows) and Eq. 5 (organ rows):
# SD of each implementation relative to the mean of all implementations.
VALUE_COL = "SD_vs_all_implementations"

# Harmonization situations to plot (left -> right within each implementation)
harmonization_order = ["No harmonization", "Bounds + Initial guess"]

harm_colors = {
    "No harmonization": "#4C72B0",
    "Bounds + Initial guess": "#DD8452",
}

# Implementations per category (order = order on the x-axis, same as the overview table)
algorithm_categories = {
    'Nonlinear LS': [
        'TCML_TechnionIIT_lsqtrf',              # 2  (1 = lsqlm excluded)
        'TCML_TechnionIIT_lsqBOBYQA',           # 4  (3 = lsq_sls_lm excluded)
        'TCML_TechnionIIT_lsq_sls_trf',         # 5
        'TCML_TechnionIIT_lsq_sls_BOBYQA',      # 6
        'ASD_MemorialSloanKettering_QAMPER_IVIM',  # 7
        'IAR_LU_biexp',                         # 8
        'OGC_AmsterdamUMC_biexp'                # 9
    ],
    'Segmented Nonlinear LS': [
        'TCML_TechnionIIT_SLS',                 # 10
        'IAR_LU_segmented_2step',               # 11
        'IAR_LU_segmented_3step',               # 12
        'IAR_LU_subtracted',                    # 13
        'OGC_AmsterdamUMC_biexp_segmented',     # 14
        'PV_MUMC_biexp',                        # 15
        'OJ_GU_seg',                            # 16
        'OJ_GU_segMATLAB'                       # 17
    ],
    'Linear LS': [
        'ETP_SRI_LinearFitting'                 # 18
    ],
    'Segmented Linear LS': [
        'TF_reference_IVIMfit',                 # 19
        'PvH_KB_NKI_IVIMfit'                    # 20
    ],
    'Variable Projection': [
        'IAR_LU_modified_mix',                  # 21
        'IAR_LU_modified_topopro'               # 22
    ],
    'Bayesian': [
        'OGC_AmsterdamUMC_Bayesian_biexp',      # 23
        'OJ_GU_bayesMATLAB'                     # 24
    ],
    'Neural network': [
        'IVIM_NEToptim',                        # 25
        'Super_IVIM_DC'                         # 26
    ],
}

category_colors = {
    "Nonlinear LS": "#1f77b4",
    "Segmented Nonlinear LS": "#9467bd",
    "Segmented Linear LS": "#d62728",
    "Variable Projection": "#ff7f0e",
    "Bayesian": "#8c564b",
    "Neural network": "#e377c2",
    "Linear LS": "#7f7f7f",
}

# Rows of the figure: (parameter name in the Excel file, y-label)
parameter_items = [
    ("D_fitted", "D\nSD (mm²/s)"),
    ("f_fitted", "f\nSD"),
    ("Dp_fitted", "D*\nSD (mm²/s)"),
]

ALL_ORGANS = "All organs"

# Marker appearance
CAT_MARKER_SIZE = 110       # SD over all organs
ORGAN_MARKER_SIZE = 50      # organ-specific SD
ORGAN_ALPHA = 0.35

# Index of each implementation in the overview table (1 and 3 are not part of the analysis).
ALGORITHM_NUMBERS = {
    "TCML_TechnionIIT_lsqtrf": 2,
    "TCML_TechnionIIT_lsqBOBYQA": 4,
    "TCML_TechnionIIT_lsq_sls_trf": 5,
    "TCML_TechnionIIT_lsq_sls_BOBYQA": 6,
    "ASD_MemorialSloanKettering_QAMPER_IVIM": 7,
    "IAR_LU_biexp": 8,
    "OGC_AmsterdamUMC_biexp": 9,
    "TCML_TechnionIIT_SLS": 10,
    "IAR_LU_segmented_2step": 11,
    "IAR_LU_segmented_3step": 12,
    "IAR_LU_subtracted": 13,
    "OGC_AmsterdamUMC_biexp_segmented": 14,
    "PV_MUMC_biexp": 15,
    "OJ_GU_seg": 16,
    "OJ_GU_segMATLAB": 17,
    "ETP_SRI_LinearFitting": 18,
    "TF_reference_IVIMfit": 19,
    "PvH_KB_NKI_IVIMfit": 20,
    "IAR_LU_modified_mix": 21,
    "IAR_LU_modified_topopro": 22,
    "OGC_AmsterdamUMC_Bayesian_biexp": 23,
    "OJ_GU_bayesMATLAB": 24,
    "IVIM_NEToptim": 25,
    "Super_IVIM_DC": 26,
}

# Layout
FIG_W, FIG_H = 22, 15
SLOT = 1.4                  # x-width reserved per harmonization situation
ORGAN_SPREAD = 0.55         # organ points are spread over +/- this value around the slot centre
BAND_GAP = 0.15             # gap between neighbouring category bands
LABEL_FONTSIZE = 13

# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_excel(INPUT_XLSX, sheet_name=SHEET)
df = df[df["Harmonization"].isin(harmonization_order)].copy()

available = set(df["Algorithm"].unique())

# Ordered list of (algorithm, category), only those present in the data
algos_ordered = [
    (a, cat)
    for cat, algos in algorithm_categories.items()
    for a in algos
    if a in available
]
missing = available - {a for a, _ in algos_ordered}
if missing:
    print("WARNING: algorithms in data but not in algorithm_categories (not plotted):")
    print(sorted(missing))

n_alg = len(algos_ordered)

alg_numbers = [ALGORITHM_NUMBERS.get(a, i + 1) for i, (a, _) in enumerate(algos_ordered)]
numbering = pd.DataFrame({
    "Number": alg_numbers,
    "Algorithm": [a for a, _ in algos_ordered],
    "Category": [c for _, c in algos_ordered],
})
numbering.to_csv("Algorithm_numbering.csv", index=False)
print(numbering.to_string(index=False))
n_harm = len(harmonization_order)
ALG_W = n_harm * SLOT

organ_markers = ["^", "s", "D", "v", "P", "X", "*", "<", ">", "h", "p", "8"]
organ_list = [o for o in df["Organ"].dropna().unique() if o != ALL_ORGANS]
organ_marker_map = {
    organ: organ_markers[i % len(organ_markers)]
    for i, organ in enumerate(organ_list)
}
organ_offsets = (
    np.linspace(-ORGAN_SPREAD, ORGAN_SPREAD, len(organ_list))
    if len(organ_list) > 1 else np.array([0.0])
)

# Fast lookup: (parameter, harmonization, algorithm, organ) -> value
lookup = (
    df.groupby(["Parameter", "Harmonization", "Algorithm", "Organ"])[VALUE_COL]
    .mean()
    .to_dict()
)


def slot_center(alg_idx, h_idx):
    return alg_idx * ALG_W + (h_idx + 0.5) * SLOT


# ============================================================
# FIGURE GEOMETRY (bands must be tall enough for the labels)
# ============================================================

label_in = LABEL_FONTSIZE * 1.4 / 72 + 0.15                  # room for the number
cat_text_in = 0.55                                            # room for category names
band_in = label_in + cat_text_in + 0.1
legend_in = 1.2

bottom = (band_in + legend_in) / FIG_H

fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(FIG_W, FIG_H), sharex=True)
fig.subplots_adjust(hspace=0.08, bottom=bottom, top=0.98, left=0.06, right=0.995)

axes_h_in = axes[-1].get_position().height * FIG_H
band_frac = band_in / axes_h_in

# ============================================================
# PLOT
# ============================================================

for ax, (param, ylabel) in zip(axes, parameter_items):

    ymax = 0

    for h_idx, harm in enumerate(harmonization_order):
        color = harm_colors[harm]

        for a_idx, (algo, cat) in enumerate(algos_ordered):
            x0 = slot_center(a_idx, h_idx)

            # ---- organ-specific (transparent, own marker per organ) ----
            for organ, off in zip(organ_list, organ_offsets):
                val = lookup.get((param, harm, algo, organ), np.nan)
                if np.isnan(val):
                    continue
                ax.scatter(
                    [x0 + off], [val],
                    s=ORGAN_MARKER_SIZE, color=color,
                    marker=organ_marker_map[organ],
                    alpha=ORGAN_ALPHA, edgecolor="none", zorder=2,
                )
                ymax = max(ymax, val)

            # ---- over all organs (solid) ----
            val_all = lookup.get((param, harm, algo, ALL_ORGANS), np.nan)
            if not np.isnan(val_all):
                ax.scatter(
                    [x0], [val_all],
                    s=CAT_MARKER_SIZE, color=color,
                    edgecolor="black", linewidth=1.2, zorder=3,
                )
                ymax = max(ymax, val_all)

    ax.set_ylim(0, ymax * 1.1)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
    ax.set_ylabel(ylabel, fontsize=16, fontweight="bold")
    ax.grid(axis="y", linestyle=":", alpha=0.4)

    # separators: thin between implementations, darker between categories
    for a_idx in range(1, n_alg):
        is_cat_boundary = algos_ordered[a_idx][1] != algos_ordered[a_idx - 1][1]
        ax.axvline(
            a_idx * ALG_W,
            color="gray" if is_cat_boundary else "lightgray",
            linewidth=1.2 if is_cat_boundary else 0.6,
            zorder=0,
        )

# ============================================================
# X AXIS: implementation names + category bands
# ============================================================

ax_b = axes[-1]
ax_b.set_xlim(0, n_alg * ALG_W)
ax_b.set_xticks([slot_center(i, 0) + SLOT / 2 for i in range(n_alg)])
ax_b.set_xticklabels(
    [str(n) for n in alg_numbers],
    rotation=0, ha="center", va="top", fontsize=LABEL_FONTSIZE,
)
for ax in axes[:-1]:
    ax.tick_params(axis="x", which="both", length=0, labelbottom=False)

trans = ax_b.get_xaxis_transform()
cats_in_order = list(dict.fromkeys(c for _, c in algos_ordered))

for cat in cats_in_order:
    idxs = [i for i, (_, c) in enumerate(algos_ordered) if c == cat]
    start = idxs[0] * ALG_W + BAND_GAP / 2
    width = len(idxs) * ALG_W - BAND_GAP

    ax_b.add_patch(
        patches.Rectangle(
            (start, -band_frac), width, band_frac,
            transform=trans,
            facecolor=category_colors.get(cat, "#cccccc"),
            alpha=0.35, edgecolor="none", clip_on=False,
        )
    )
    ax_b.text(
        start + width / 2, -band_frac + 0.08 / axes_h_in,
        textwrap.fill(cat, 14),
        transform=trans, ha="center", va="bottom",
        fontsize=11, fontweight="bold", clip_on=False,
    )

# ============================================================
# LEGEND
# ============================================================

legend_handles = [
    Line2D([0], [0], marker="o", linestyle="", markersize=11,
           markerfacecolor=harm_colors[h], markeredgecolor="black", label=h)
    for h in harmonization_order
]
legend_handles += [
    Line2D([0], [0], marker="o", linestyle="", markersize=11,
           markerfacecolor="gray", markeredgecolor="black", label="All organs"),
]
legend_handles += [
    Line2D([0], [0], marker=organ_marker_map[o], linestyle="", markersize=9,
           markerfacecolor="gray", markeredgecolor="none", alpha=0.6, label=o)
    for o in organ_list
]

fig.legend(
    handles=legend_handles,
    loc="lower center",
    bbox_to_anchor=(0.5, 0.0),
    ncol=4,
    frameon=False,
    fontsize=14,
)

filename = f"SD_per_implementation_SNR{SNR}_fullbvals.png"
plt.savefig(filename, dpi=300, bbox_inches="tight")
plt.close()

print(f"Saved {filename}")