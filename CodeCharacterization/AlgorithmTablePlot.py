"""
Script to plot a table with all implementations and their category.
"""
import matplotlib.pyplot as plt
import pandas as pd

# ============================================================
# DATA (from your table)
# ============================================================
data = [
    ("Nonlinear LS", 1, "TCML_TechnionIIT_lsqlm", "Python"),
    ("Nonlinear LS", 2, "TCML_TechnionIIT_lsqtrf", "Python"),
    ("Nonlinear LS", 3, "TCML_TechnionIIT_lsq_sls_lm", "Python"),
    ("Nonlinear LS", 4, "TCML_TechnionIIT_lsqBOBYQA", "Python"),
    ("Nonlinear LS", 5, "TCML_TechnionIIT_lsq_sls_trf", "Python"),
    ("Nonlinear LS", 6, "TCML_TechnionIIT_lsq_sls_BOBYQA", "Python"),
    ("Nonlinear LS", 7, "ASD_MemorialSloanKettering_QAMPER_IVIM", "Matlab"),
    ("Nonlinear LS", 8, "IAR_LU_biexp", "Python"),
    ("Nonlinear LS", 9, "OGC_AmsterdamUMC_biexp", "Python"),

    ("Variable projection", 10, "IAR_LU_modified_mix", "Python"),
    ("Variable projection", 11, "IAR_LU_modified_topopro", "Python"),

    ("Linear LS", 12, "ETP_SRI_LinearFitting", "Python"),

    ("Segmented linear LS", 13, "TF_reference_IVIMfit", "Python"),
    ("Segmented linear LS", 14, "PvH_KB_NKI_IVIMfit", "Python"),

    ("Segmented nonlinear LS", 15, "TCML_TechnionIIT_SLS", "Python"),
    ("Segmented nonlinear LS", 16, "IAR_LU_segmented_2step", "Python"),
    ("Segmented nonlinear LS", 17, "IAR_LU_segmented_3step", "Python"),
    ("Segmented nonlinear LS", 18, "IAR_LU_subtracted", "Python"),
    ("Segmented nonlinear LS", 19, "OGC_AmsterdamUMC_biexp_segmented", "Python"),
    ("Segmented nonlinear LS", 20, "PV_MUMC_biexp", "Python"),
    ("Segmented nonlinear LS", 21, "OJ_GU_seg", "Python"),
    ("Segmented nonlinear LS", 22, "OJ_GU_segMATLAB", "Matlab"),

    ("Bayesian", 23, "OGC_AmsterdamUMC_Bayesian_biexp", "Python"),
    ("Bayesian", 24, "OJ_GU_bayesMATLAB", "Matlab"),

    ("Neural network", 25, "IVIM_NEToptim", "Python"),
    ("Neural network", 26, "Super_IVIM_DC", "Python"),
]

df = pd.DataFrame(data, columns=[
    "Implementation category", "Index", "Algorithm name", "Language"
])

# ============================================================
# CATEGORY COLORS (consistent with grouping logic)
# ============================================================
category_colors = {
    "Nonlinear LS": "#c6dbef",
    "Variable projection": "#fdd0a2",
    "Linear LS": "#c7e9c0",
    "Segmented linear LS": "#f7f7b6",
    "Segmented nonlinear LS": "#fcbba1",
    "Bayesian": "#dadaeb",
    "Neural network": "#e6f5c9"
}

# ============================================================
# FIGURE
# ============================================================
import matplotlib.pyplot as plt
import pandas as pd

# (same df + category_colors as before)

fig, ax = plt.subplots(figsize=(16, 10))
ax.axis('off')

table = ax.table(
    cellText=df.values,
    colLabels=df.columns,
    cellLoc='center',
    loc='center'
)

# ============================================================
# FONT + SCALE
# ============================================================
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1, 1.6)

# ============================================================
# COLUMN WIDTH FIX (IMPORTANT PART)
# ============================================================
col_widths = [0.25, 0.08, 0.55, 0.12]  # category, index, name, language

for i in range(len(df) + 1):  # +1 includes header
    for j, w in enumerate(col_widths):
        table[i, j].set_width(w)

# ============================================================
# HEADER STYLE
# ============================================================
for j in range(len(df.columns)):
    cell = table[0, j]
    cell.set_facecolor("#404040")
    cell.get_text().set_color("white")
    cell.get_text().set_weight("bold")

# ============================================================
# ROW COLORS
# ============================================================
for i in range(len(df)):
    cat = df.iloc[i]["Implementation category"]
    color = category_colors.get(cat, "white")

    for j in range(len(df.columns)):
        table[i + 1, j].set_facecolor(color)

# ============================================================
# SAVE
# ============================================================
plt.tight_layout()

plt.show()